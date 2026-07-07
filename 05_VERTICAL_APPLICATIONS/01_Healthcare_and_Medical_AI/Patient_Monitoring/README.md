# Patient Monitoring AI

This directory covers AI for patient monitoring, early warning, remote monitoring, ICU analytics, wearables, and clinical alerting.

## Scope

- Vital signs, waveform data, labs, EHR events, wearables, device streams, deterioration prediction, sepsis warning, and alarm triage.
- Track sampling rate, label timing, alert threshold, lead time, false-alarm burden, calibration, and clinical workflow.

## Reference Links

- PhysioNet: https://physionet.org/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-IV Waveform Database: https://physionet.org/content/mimic4wdb/
- HL7 FHIR: https://hl7.org/fhir/

## Routing Rules

- Put ICU-specific monitoring in `ICU/`.
- Put edge deployment in `../Edge_AI/`.
- Put clinical decision support in CDS directories.
