# Medical Prompts

This directory covers prompts for healthcare, medicine, biomedical research, clinical operations, and patient-facing workflows.

## Scope

- Clinical summarization, documentation support, triage support, patient education, coding assistance, medical literature review, and biomedical extraction prompts.
- Safety boundaries for diagnosis, treatment, medical advice, uncertainty, citations, protected health information, and clinician review.
- Prompt templates for medical QA, report structuring, discharge summaries, trial matching, and patient communication.

## Reference Links

- OpenAI prompt engineering: https://developers.openai.com/api/docs/guides/prompt-engineering
- Anthropic prompt engineering: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- FDA AI/ML in SaMD: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device
- WHO AI for health ethics: https://www.who.int/publications/i/item/9789240029200
- HL7 FHIR: https://hl7.org/fhir/

## Routing Rules

- Put clinical NLP models in `../../../05_VERTICAL_APPLICATIONS/01_Healthcare_and_Medical_AI/Clinical_NLP/`.
- Put healthcare model catalogs in medical AI model directories.
- Put prompt-safety tests in evaluation or MLOps directories.
