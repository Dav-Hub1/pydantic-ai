import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model_name = os.getenv("TRIAGE_MODEL", "openai:gpt-4-nano-2024-06-01")  # Default to a specific model if not set in .env

# Mock database for demonstration purposes
@dataclass
class PatientRecord:
    patient_id: str
    name: str
    age: int
    medical_history: Dict[str, Any]

PatientRecords_DB = {
    "12345": PatientRecord(
        patient_id="12345",
        name="John Doe",
        age=30,
        medical_history={"allergies": ["penicillin"], "conditions": ["hypertension"]},
    ),
    "67890": PatientRecord(
        patient_id="67890",
        name="Jane Smith",
        age=25,
        medical_history={"allergies": [], "conditions": ["asthma"]},
    ),
}

class PatientDataRetriever:
    
    async def get_patient_record(self, patient_id: int) -> str:
        # Simulate an asynchronous database call
        patient = PatientRecords_DB.get(str(patient_id))    
        await asyncio.sleep(0.1)  # Simulating I/O delay
        return patient.name if patient else "Patient not found"
    
    async def get_patient_medical_history(self, patient_id: int) -> Dict[str, Any]:
        # Simulate an asynchronous database call
        patient = PatientRecords_DB.get(str(patient_id))    
        await asyncio.sleep(0.1)  # Simulating I/O delay
        return patient.medical_history if patient else {"error": "Patient not found"}

@dataclass
class TriageDependencies:
    patient_id: int
    patient_data_retriever: PatientDataRetriever

class TriageOutput(BaseModel):
    response_message: str = Field(..., description="Message to be sent to the patient")
    escalation_needed: bool = Field(..., description="Indicates if escalation to a healthcare provider is needed")
    urgency_level: Optional[str] = Field(None, description="The urgency level of the patient's condition")

triage_agent = Agent[TriageDependencies, TriageOutput](
    model=model_name,
    output_type=TriageOutput,
    system_prompt=(
        """You are a medical triage assistant.
        Your task is to evaluate the patient's condition based on their medical history and current symptoms,
        and determine the appropriate response message, whether escalation to a healthcare provider is needed,
        and the urgency level of their condition.
        
        Always consider the patient's medical history when evaluating their symptoms. If the patient has a history of severe conditions, be more cautious in your response."""
    ),
    deps_type=TriageDependencies,

)

@triage_agent.system_prompt
async def add_patient_data(
    ctx: RunContext[TriageDependencies],
) -> str:
    patient_id = ctx.deps.patient_id
    retriever = ctx.deps.patient_data_retriever
    
    # Retrieve patient data
    patient_name = await retriever.get_patient_record(patient_id)
    medical_history = await retriever.get_patient_medical_history(patient_id)
    
    # Format the patient data for the prompt
    patient_data_str = f"Patient Name: {patient_name}\nMedical History: {medical_history}"
    
    return patient_data_str

@triage_agent.tool
async def get_allergies(
    ctx: RunContext[TriageDependencies]) -> Dict[str, Any]:
    # This tool can be used to perform additional checks or calculations based on the patient's data
    # For demonstration, we'll just return a placeholder response
    return await ctx.deps.patient_data_retriever.get_patient_medical_history(ctx.deps.patient_id)

async def main() -> None:
    deps = TriageDependencies(
        patient_id=12345,
        patient_data_retriever=PatientDataRetriever(),
    )
    result = await triage_agent.run(
        "Patient is experiencing chest pain and shortness of breath.",
        deps=deps,
    )
    
    print(result.output)
    
if __name__ == "__main__":
    asyncio.run(main())