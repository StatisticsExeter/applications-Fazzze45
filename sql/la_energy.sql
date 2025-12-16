WITH raw AS (
    SELECT 
        co2_emissions_current AS shortfall,
        construction_age_estimated AS age,
        heating_cost_current AS n_rooms,
        local_authority AS local_authority_code
    FROM energy.energy_certificates
    WHERE 
        co2_emissions_current IS NOT NULL
        AND construction_age_estimated IS NOT NULL
        AND heating_cost_current IS NOT NULL
        AND local_authority IS NOT NULL
)
SELECT 
    shortfall,
    age,
    n_rooms,
    local_authority_code
FROM raw
WHERE shortfall > 0 AND n_rooms > 0
LIMIT 500;