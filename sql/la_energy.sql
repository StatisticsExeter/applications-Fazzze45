WITH raw AS (
    SELECT
        co2_emissions_current AS shortfall,
        construction_age_estimated AS age,
        heating_cost_current AS n_rooms,
        local_authority AS local_authority_code,
        ROW_NUMBER() OVER (
            PARTITION BY local_authority
            ORDER BY lodgement_datetime DESC
        ) AS row_num
    FROM energy.energy_certificates
    WHERE
        co2_emissions_current IS NOT NULL
        AND construction_age_estimated IS NOT NULL
        AND heating_cost_current IS NOT NULL
        AND local_authority IS NOT NULL
        AND co2_emissions_current > 0
        AND heating_cost_current > 0
)
SELECT
    shortfall,
    age,
    n_rooms,
    local_authority_code
FROM raw
WHERE row_num <= 2
LIMIT 500;
