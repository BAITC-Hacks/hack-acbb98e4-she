"""Независимая предварительная проверка финального плана по публичным данным.

Аудит не импортирует агент, среду или скоринг и не оценивает финансовый эффект.
Передавайте остатки бюджета и контактов ПОСЛЕ пилотов. Суммы относятся только
к финальным кампаниям; повторные контакты расходуют лимиты повторно.

Проверка намеренно строже скоринга: аудитория > 5000, пустой сегмент и выход за
остатки ресурсов считаются ошибкой, а не исправляются усечением аудитории.
Пропущенный фильтр, None и NaN означают отсутствие фильтра. Пустая строка —
ошибка. В официальном пакете только filter_current_tariff допускает список
через ';'; остальные фильтры принимают одно значение сегмента.

При ошибке схемы показатели такой кампании равны нулю: сводные суммы тогда
не описывают полный план и не должны использоваться до исправления errors.
"""

import math
from collections.abc import Mapping
from numbers import Real

import pandas as pd


_FILTER_COLUMNS = {
    "filter_arpu_segment": "arpu_segment",
    "filter_data_segment": "data_segment",
    "filter_call_segment": "call_segment",
    "filter_current_tariff": "current_tariff",
}
_SEGMENT_VALUES = {
    "filter_arpu_segment": {"LOW", "MID", "HIGH"},
    "filter_data_segment": {"NON_USER", "LITE", "HEAVY"},
    "filter_call_segment": {"LOW", "MEDIUM", "HIGH"},
}
_ALLOWED_FIELDS = {"campaign_name", "target_tariff", "channel"} | set(_FILTER_COLUMNS)
_PROFILE_COLUMNS = {"ID_NUMBER"} | set(_FILTER_COLUMNS.values())


def _is_missing(value):
    if value is None:
        return True
    if not isinstance(value, Real):
        return False
    try:
        return math.isnan(float(value))
    except (OverflowError, ValueError):
        return False


def _nonnegative_number(value, integer=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and number >= 0 and (not integer or number.is_integer())


def audit_plan(campaigns, profile, tariffs, channels, remaining_budget, remaining_contacts):
    """Вернуть JSON-совместимый отчёт; входные данные не изменяются.

    valid=True означает лишь соблюдение проверяемых формальных ограничений.
    Это не прогноз прибыли и не результат закрытой проверки организаторов.
    """
    result = {
        "valid": False,
        "errors": [],
        "total_contacts": 0,
        "unique_contacts": 0,
        "communication_cost": 0.0,
        "campaigns": [],
    }
    errors = result["errors"]
    if not isinstance(campaigns, list):
        errors.append("План должен быть списком словарей кампаний.")
        return result
    if not 1 <= len(campaigns) <= 10:
        errors.append("В финальном плане должно быть от 1 до 10 кампаний.")

    context_errors = []
    if not isinstance(profile, pd.DataFrame):
        context_errors.append("profile должен быть pandas.DataFrame.")
    elif not profile.columns.is_unique:
        context_errors.append("В profile повторяются имена колонок.")
    else:
        missing_columns = sorted(_PROFILE_COLUMNS - set(profile.columns))
        if missing_columns:
            context_errors.append("В profile отсутствуют колонки: " + ", ".join(missing_columns) + ".")
        elif profile["ID_NUMBER"].isna().any() or profile["ID_NUMBER"].duplicated().any():
            context_errors.append("ID_NUMBER в profile должны быть уникальными и непустыми.")
    if not isinstance(tariffs, pd.DataFrame) or "tariff_plan_code" not in tariffs.columns:
        context_errors.append("tariffs должен содержать колонку tariff_plan_code.")
    elif not tariffs.columns.is_unique:
        context_errors.append("В tariffs повторяются имена колонок.")
    if not isinstance(channels, Mapping) or not channels:
        context_errors.append("channels должен быть непустым словарём доступных каналов.")
    if not _nonnegative_number(remaining_budget):
        context_errors.append("remaining_budget должен быть конечным неотрицательным числом после пилотов.")
    if not _nonnegative_number(remaining_contacts, integer=True):
        context_errors.append("remaining_contacts должен быть неотрицательным целым числом после пилотов.")
    if context_errors:
        errors.extend(context_errors)
        return result

    known_tariffs = set(tariffs["tariff_plan_code"].dropna().astype(str))
    all_ids = set()
    costs = []
    for index, campaign in enumerate(campaigns, start=1):
        detail = {
            "index": index,
            "campaign_name": "campaign_" + str(index),
            "target_tariff": None,
            "channel": None,
            "contacts": 0,
            "unique_contacts": 0,
            "communication_cost": 0.0,
            "valid": False,
            "errors": [],
        }
        result["campaigns"].append(detail)
        campaign_errors = detail["errors"]
        if not isinstance(campaign, dict):
            campaign_errors.append("Ожидался словарь кампании.")
        else:
            unknown = set(campaign) - _ALLOWED_FIELDS
            if unknown:
                campaign_errors.append("Недопустимые поля: " + ", ".join(sorted(map(str, unknown))) + ".")
            if "campaign_name" in campaign:
                if isinstance(campaign["campaign_name"], str):
                    detail["campaign_name"] = campaign["campaign_name"]
                else:
                    campaign_errors.append("campaign_name должен быть строкой, если он указан.")

            target = campaign.get("target_tariff")
            channel = campaign.get("channel")
            if not isinstance(target, str) or target not in known_tariffs:
                campaign_errors.append("target_tariff отсутствует или не найден в tariffs.")
            else:
                detail["target_tariff"] = target
            cost_per_contact = None
            if not isinstance(channel, str) or channel not in channels:
                campaign_errors.append("channel отсутствует или не найден в channels.")
            else:
                detail["channel"] = channel
                config = channels[channel]
                cost_per_contact = config.get("cost_per_contact") if isinstance(config, Mapping) else None
                if not _nonnegative_number(cost_per_contact):
                    campaign_errors.append("Стоимость cost_per_contact для канала должна быть конечной и неотрицательной.")

            parsed_filters = {}
            for field in _FILTER_COLUMNS:
                value = campaign.get(field)
                if _is_missing(value):
                    continue
                if not isinstance(value, str) or not value.strip():
                    campaign_errors.append(field + ": нужна непустая строка, None/NaN или отсутствие поля.")
                    continue
                if field == "filter_current_tariff":
                    values = [item.strip() for item in value.split(";")]
                    if any(not item for item in values):
                        campaign_errors.append(field + ": пустое значение между разделителями ';'.")
                    elif set(values) - known_tariffs:
                        campaign_errors.append(field + ": неизвестные тарифы " + ", ".join(sorted(set(values) - known_tariffs)) + ".")
                    else:
                        parsed_filters[field] = values
                elif value not in _SEGMENT_VALUES[field]:
                    campaign_errors.append(field + ": требуется одно из значений " + ", ".join(sorted(_SEGMENT_VALUES[field])) + ".")
                else:
                    parsed_filters[field] = [value]

            if not campaign_errors:
                mask = pd.Series(True, index=profile.index)
                for field, values in parsed_filters.items():
                    mask &= profile[_FILTER_COLUMNS[field]].isin(values)
                ids = set(profile.loc[mask, "ID_NUMBER"].tolist())
                count = int(mask.sum())
                cost = float(count * float(cost_per_contact))
                detail["contacts"] = count
                detail["unique_contacts"] = len(ids)
                detail["communication_cost"] = cost if math.isfinite(cost) else None
                result["total_contacts"] += count
                if math.isfinite(cost):
                    costs.append(cost)
                else:
                    campaign_errors.append("Затраты на коммуникацию слишком велики для числового отчёта.")
                all_ids.update(ids)
                if count == 0:
                    campaign_errors.append("Фильтры выбрали пустую аудиторию.")
                elif count > 5000:
                    campaign_errors.append("Аудитория " + str(count) + " превышает лимит 5000: сузьте фильтры.")

        detail["valid"] = not campaign_errors
        errors.extend("Кампания " + str(index) + ": " + error for error in campaign_errors)

    result["unique_contacts"] = len(all_ids)
    try:
        result["communication_cost"] = float(math.fsum(costs))
    except OverflowError:
        result["communication_cost"] = None
        errors.append("Суммарные затраты слишком велики для числового отчёта.")
    if any(item["communication_cost"] is None for item in result["campaigns"]):
        result["communication_cost"] = None
    if result["total_contacts"] > remaining_contacts:
        errors.append("Нужно " + str(result["total_contacts"]) + " контактов, после пилотов осталось " + str(int(remaining_contacts)) + ".")
    if result["communication_cost"] is not None and result["communication_cost"] > remaining_budget:
        errors.append("Затраты на коммуникацию " + str(result["communication_cost"]) + " превышают остаток бюджета после пилотов " + str(float(remaining_budget)) + ".")
    result["valid"] = not errors
    return result
