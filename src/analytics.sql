-- QUERY: data_provenance_audit
SELECT
    l.dataset_key,
    l.dataset_name,
    l.ingestion_mode,
    l.clean_rows,
    l.rows_2025_plus,
    l.earliest_observation,
    l.latest_observation,
    l.retrieved_at_utc
FROM data_ingestion_log l
ORDER BY l.dataset_key;

-- QUERY: sf_weekly_trends_and_wow
WITH ordered_weekly AS (
    SELECT
        week_start_date,
        week_end_date,
        year,
        month,
        epi_week,
        respiratory_virus,
        new_admissions,
        admission_rate_per_100k,
        activity_level,
        trend,
        is_2025_plus,
        LAG(new_admissions, 1) OVER (
            PARTITION BY respiratory_virus
            ORDER BY week_start_date
        ) AS prev_week_admissions,
        ROUND(
            AVG(new_admissions) OVER (
                PARTITION BY respiratory_virus
                ORDER BY week_start_date
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ), 2
        ) AS rolling_4wk_avg_admissions
    FROM sf_respiratory_weekly
)
SELECT
    week_start_date,
    week_end_date,
    year,
    month,
    epi_week,
    respiratory_virus,
    new_admissions,
    prev_week_admissions,
    CASE
        WHEN prev_week_admissions IS NULL OR prev_week_admissions = 0 THEN NULL
        ELSE ROUND(((new_admissions - prev_week_admissions) * 100.0) / prev_week_admissions, 2)
    END AS wow_pct_change,
    rolling_4wk_avg_admissions,
    admission_rate_per_100k,
    activity_level,
    trend
FROM ordered_weekly
WHERE is_2025_plus = 1
ORDER BY week_start_date DESC, respiratory_virus ASC;

-- QUERY: sf_monthly_disease_summary
SELECT
    year_month,
    respiratory_virus,
    COUNT(*) AS weeks_reported,
    SUM(new_admissions) AS total_monthly_admissions,
    ROUND(AVG(new_admissions), 2) AS avg_weekly_admissions,
    MAX(new_admissions) AS peak_weekly_admissions,
    ROUND(AVG(admission_rate_per_100k), 2) AS avg_rate_per_100k
FROM sf_respiratory_weekly
WHERE is_2025_plus = 1
GROUP BY year_month, respiratory_virus
ORDER BY year_month DESC, total_monthly_admissions DESC;

-- QUERY: sf_disease_share_and_ranking
WITH virus_totals AS (
    SELECT
        respiratory_virus,
        COUNT(*) AS weeks_observed,
        SUM(new_admissions) AS total_admissions_2025_plus,
        ROUND(AVG(new_admissions), 2) AS mean_weekly_admissions,
        MAX(new_admissions) AS max_weekly_admissions,
        ROUND(AVG(admission_rate_per_100k), 2) AS mean_rate_per_100k,
        MAX(admission_rate_per_100k) AS peak_rate_per_100k
    FROM sf_respiratory_weekly
    WHERE is_2025_plus = 1
    GROUP BY respiratory_virus
),
grand_total AS (
    SELECT SUM(total_admissions_2025_plus) AS all_virus_admissions
    FROM virus_totals
)
SELECT
    v.respiratory_virus,
    v.weeks_observed,
    v.total_admissions_2025_plus,
    ROUND((v.total_admissions_2025_plus * 100.0) / g.all_virus_admissions, 2) AS pct_share_of_admissions,
    v.mean_weekly_admissions,
    v.max_weekly_admissions,
    v.mean_rate_per_100k,
    v.peak_rate_per_100k,
    RANK() OVER (ORDER BY v.total_admissions_2025_plus DESC) AS burden_rank
FROM virus_totals v
CROSS JOIN grand_total g
ORDER BY burden_rank;

-- QUERY: sf_activity_level_distribution
SELECT
    respiratory_virus,
    activity_level,
    trend,
    COUNT(*) AS week_count,
    ROUND(AVG(new_admissions), 2) AS avg_admissions_in_state
FROM sf_respiratory_weekly
WHERE is_2025_plus = 1
GROUP BY respiratory_virus, activity_level, trend
ORDER BY respiratory_virus, week_count DESC;

-- QUERY: nhs_dimension_overview
SELECT
    org_level,
    dimension,
    count_of,
    COUNT(*) AS record_count,
    COUNT(DISTINCT org_code) AS unique_orgs,
    ROUND(SUM(COALESCE(measure_value, 0)), 0) AS sum_measure_value
FROM nhs_provider_activity
GROUP BY org_level, dimension, count_of
ORDER BY record_count DESC;

-- QUERY: nhs_top_providers_activity
WITH provider_summary AS (
    SELECT
        org_code,
        org_name,
        count_of,
        SUM(COALESCE(measure_value, 0)) AS total_activity,
        SUM(COALESCE(measure_value_0_18, 0)) AS age_0_18_activity,
        SUM(COALESCE(measure_value_19_64, 0)) AS age_19_64_activity,
        SUM(COALESCE(measure_value_65_plus, 0)) AS age_65_plus_activity
    FROM nhs_provider_activity
    WHERE org_level = 'Provider'
      AND dimension = 'ServiceType'
      AND count_of = 'UniqueCareContacts'
    GROUP BY org_code, org_name, count_of
)
SELECT
    org_code,
    org_name,
    total_activity,
    age_0_18_activity,
    age_19_64_activity,
    age_65_plus_activity,
    CASE
        WHEN total_activity > 0 THEN ROUND((age_65_plus_activity * 100.0) / total_activity, 2)
        ELSE 0.0
    END AS pct_age_65_plus,
    RANK() OVER (ORDER BY total_activity DESC) AS provider_rank
FROM provider_summary
WHERE total_activity > 0
ORDER BY provider_rank
LIMIT 15;

-- QUERY: nhs_service_type_demographics
WITH service_agg AS (
    SELECT
        measure AS service_code,
        measure_desc AS service_type,
        COUNT(DISTINCT org_code) AS active_providers,
        SUM(COALESCE(measure_value, 0)) AS total_contacts,
        SUM(COALESCE(measure_value_0_18, 0)) AS contacts_0_18,
        SUM(COALESCE(measure_value_19_64, 0)) AS contacts_19_64,
        SUM(COALESCE(measure_value_65_plus, 0)) AS contacts_65_plus
    FROM nhs_provider_activity
    WHERE org_level = 'Provider'
      AND dimension = 'ServiceType'
      AND count_of = 'UniqueCareContacts'
      AND measure_desc != 'Unspecified'
    GROUP BY measure, measure_desc
)
SELECT
    service_code,
    service_type,
    active_providers,
    total_contacts,
    contacts_0_18,
    contacts_19_64,
    contacts_65_plus,
    CASE
        WHEN total_contacts > 0 THEN ROUND((contacts_65_plus * 100.0) / total_contacts, 2)
        ELSE 0.0
    END AS pct_65_plus,
    CASE
        WHEN total_contacts > 0 THEN ROUND((contacts_0_18 * 100.0) / total_contacts, 2)
        ELSE 0.0
    END AS pct_0_18
FROM service_agg
WHERE total_contacts > 0
ORDER BY total_contacts DESC
LIMIT 15;