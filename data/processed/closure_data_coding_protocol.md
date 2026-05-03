# School Closure Data Coding Protocol

## Purpose

The closure data measure actual instructional disruption. In this project, hurricane-related closure days are treated as the main treatment-intensity variable in the school-closure analysis.

They are not pre-treatment controls. They occur after the disaster shock and may mediate the relationship between physical hurricane exposure and TOPS eligibility.

## Raw unit

The raw unit is a closure announcement or closure record.

Examples:

- A district announcement that all schools are closed from Monday to Wednesday.
- A news article reporting that a parish school system closed due to Hurricane Ida.
- A school-specific notice that one high school closed due to storm damage.

## Preferred cleaned unit

The preferred cleaned unit is a closure record with start and end date, closure scope, parish/district/school identifiers, and reason category.

Later scripts will expand these records to school-date level and aggregate to NCES school × graduation year.

## Academic-year mapping

For graduation year t, the relevant school year is:

- August 1 of t-1 through May 31 of t.

Example:

- graduation_year = 2021
- closure window = 2020-08-01 to 2021-05-31

Closures outside this window should not be counted for that graduating cohort in the main specification.

## Closure scope

Use one of:

- `school_specific`
- `district_system`
- `parishwide`
- `statewide`
- `unclear`

Coding rules:

- If the source says all schools in a parish/district are closed, code as `district_system` or `parishwide`.
- If the source names a specific school, code as `school_specific`.
- If the source only mentions a general emergency declaration, do not count it as a school closure unless school closure is explicitly stated.

## Closure reason category

Use one of:

- `hurricane`
- `tropical_storm`
- `severe_weather`
- `flood`
- `power_outage`
- `covid`
- `other`
- `unclear`

Main treatment uses:

- `hurricane_related == True`
- `instructional_closure == True`
- `include_in_main_closure_measure == True`

## Hurricane-related coding

Set `hurricane_related = True` if the closure is explicitly related to:

- a named hurricane,
- a named tropical storm,
- hurricane landfall,
- tropical storm conditions,
- hurricane evacuation,
- hurricane-related flooding, damage, or power outage.

Set `hurricane_related = False` for:

- COVID closure,
- normal holiday,
- teacher work day,
- unrelated illness,
- ordinary maintenance,
- non-weather administrative closure.

Set `needs_manual_review = True` if the source is ambiguous.

## Instructional closure

Set `instructional_closure = True` if students lost ordinary in-person instructional time.

Set `instructional_closure = False` if:

- the source only reports office closure,
- extracurricular cancellation only,
- school was open virtually and no instructional loss is indicated,
- the record is only an emergency declaration with no school closure.

## Main closure measure

The main treatment variable will be:

`closure_days_hurricane_related`

At the school-year level, this is the number of instructional closure days during the academic-year window that are hurricane-related.

Additional variables:

- `closure_any_hurricane_related`
- `closure_event_count_hurricane_related`
- `max_consecutive_closure_days_hurricane_related`

## Source hierarchy

Preferred sources:

1. Official school/district/parish school-system announcements.
2. CDC official school closure data when available and clearly identifiable.
3. Local news reports explicitly describing school closures.
4. GOHSEP / emergency information only as contextual evidence, not as closure proof.

## API use

OpenRouter / LLM extraction may be used only to convert source text into structured records.

The API should not invent dates, schools, parishes, or closure reasons. If the text is ambiguous, the record must be flagged for manual review.
