from langchain.prompts import PromptTemplate


def get_prompt(question_data):
    question_type = question_data.get("question_type", "")
    question = question_data.get("question", "")

    if question_type == "single_choice":
        options_text = ""
        # ensure options key exists and is a dictionary
        options = question_data.get("options", {})
        for letter, option in options.items():
            options_text += f"\n{letter}. {option}"
        
        prompt_template = PromptTemplate.from_template(
            """You are a biologist. Please carefully read the provided instruction and answer the questions.
Question: {question}
Options: {options_text}
Task Instruction: Only provide the correct option item (e.g., 'A'). Do not include option details and any additional information or explanations."""
        )
        
        return prompt_template.format(question=question, options_text=options_text)
    
    elif question_type == "multiple_choice":
        options_text = ""
        options = question_data.get("options", {})
        for letter, option in options.items():
            options_text += f"\n{letter}. {option}"
        prompt_template = PromptTemplate.from_template(
            """You are a biologist. Please carefully read the provided instruction and answer the questions.
Question: {question}
Options: {options_text}
Task Instruction:  Only provide the letters of the correct options, separated by commas (e.g., 'A, B'). Do not include option details and any additional information or explanations."""
        )
        return prompt_template.format(question=question, options_text=options_text)
    
    elif question_type == "designation":
        prompt_template = PromptTemplate.from_template(
            """You are a biologist. Please carefully read the provided instruction and answer the questions.
Question: {question}
Task Instruction:  
Only provide all related proteins of the given gene in a list format and format them as a JSON dictionary, like {{"designation": ["protein1", "protein2"]}}. If you cannot answer the question, response an empty JSON dictionary with {{"designation": []}}."""
        )
        return prompt_template.format(question=question)
    
    elif question_type == "expression":
        prompt_template = PromptTemplate.from_template(
            """You are a biologist. Please carefully read the provided instruction and answer the questions.
Question: {question}
Task Instruction:  
Provide a JSON dictionary with two keys: 'Tissue' and 'Category'.
'Tissue' key's value should be a list of relevant tissues.
'Category' key's value must be one of the following categories:
'Low expression', 'Broad expression', 'Biased expression', 'Restricted expression', 'Ubiquitous expression'.
Here are the candidate tissues:
'fat', 'brain', 'gall bladder', 'heart', 'endometrium', 'prostate', 'testis', 'salivary gland', 'lung', 
'ovary', 'esophagus', 'colon', 'duodenum', 'liver', 'lymph node', 'appendix', 'small intestine', 'skin', 
'urinary bladder', 'stomach', 'spleen', 'adrenal', 'kidney', 'placenta', 'bone marrow', 'thyroid', 'pancreas'.
For 'Low expression' category, the tissue list should be 'Low expression'. For example: {{'Tissue': ['Low expression'], 'Category': 'Low expression'}}.
For example of other categories: {{'Tissue': ['liver', 'colon'], 'Category': 'Biased expression'}}.
Only provide the JSON dictionary as your response, without any additional text or explanations."""
        )
        return prompt_template.format(question=question)
    
    elif question_type == "go_annotation":
        # Base prompt for both cases (with or without context)
        base_prompt = """You are a biologist. Please carefully read the provided instruction and answer the questions.
    Question: {question}
    Task Instruction:  
    Provide a JSON array of objects. Each object in the array must contain two keys: 'go' and 'evidence'.
    The value for 'go' should be the full Gene Ontology annotation.
    The value for 'evidence' must be one of the following codes:
    EXP, IDA, IPI, IMP, IGI, IEP, IBA, IBD, IKR, IRD, ISM, IGC, RCA, TAS, NAS, IC, ND, IEA.

    For example:
    [{{"go": "located in nucleus", "evidence": "IDA"}}, {{"go": "calcium ion binding", "evidence": "IEA"}}]

    Do not include any additional information, explanations, or text before or after the JSON array.
    """
        context = question_data.get("context", None)  

        if context:
            prompt_template = PromptTemplate.from_template(
                f"{base_prompt}\n"
                f"Your answer should based on the context.\n"
                f"Context: {context}\n"
            )
            return prompt_template.format(question=question)
        else:
            # For questions without context, just use the base prompt
            prompt_template = PromptTemplate.from_template(base_prompt)
            return prompt_template.format(question=question)
        
    elif question_type == "summary":
        # Get context, default to None if not present
        context = question_data.get("context", None)
        question = question_data.get("question")

        # Define the core template for the task
        task_instruction_template = """
        You are a biologist. Please carefully read the provided instruction and answer the questions.
        Question: {question}
        Task Instruction:
        Provide a functional summary of the given gene.
        Do not include any additional information, explanations, or text before or after the summary.
        """

        if context:
            # Add context-specific instructions to the core template
            full_prompt = f"""{task_instruction_template}
            You could use the following context to answer the question.
            Context: {context}
            """
            prompt_template = PromptTemplate.from_template(full_prompt)
            return prompt_template.format(question=question)
        else:
            # Use the core template directly when there's no context
            prompt_template = PromptTemplate.from_template(task_instruction_template)
            return prompt_template.format(question=question)
        
    return question


# ==============
# Pydantic models for question types
# ==============

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, RootModel
from langchain.output_parsers import PydanticOutputParser

# Common literal set for options A-D
Option = Literal["A","B","C","D"]

class SingleChoiceSchema(BaseModel):
    """Single Choice: reformat to single uppercase letter"""
    answer: Option = Field(..., description="The selected option, must be one of A-D")

class MultipleChoiceSchema(BaseModel):
    """Multiple Choice: reformat to multiple uppercase letters, sorted in alphabetical order and without duplicates"""
    answers: List[Option] = Field(..., description="A list of uppercase letters for multiple options, sorted in alphabetical order and without duplicates, limited in A-D")

class DesignationSchema(BaseModel):
    """Designation: normalize gene/entity names to a standard format"""
    designation: List[str] = Field(..., description="A list of normalized designations")

class ExpressionSchema(BaseModel):
    """Expression: output a dictionary with keys 'Tissue' (list[str]) and 'Category' (str)"""
    Tissue: List[str] = Field(..., description="A list of tissue names, such as ['brain', 'testis']")
    Category: str = Field(..., description="A category of expression, such as 'Broad expression' or 'Biased expression'")


class GOItem(BaseModel):
    go: str = Field(..., description="GO Term, such as 'enables GTPase activity'")
    evidence: str = Field(..., description="Evidence code, such as 'IDA', 'IMP', etc.")

class GOAnnotationSchema(RootModel[List[GOItem]]):
    pass

class SummarySchema(BaseModel):
    """Summary: answer a concise summary text"""
    summary: str = Field(..., description="A normalized summary")

def get_parser_by_type(question_type: str) -> PydanticOutputParser:
    mapping = {
        "single_choice": PydanticOutputParser(pydantic_object=SingleChoiceSchema),
        "multiple_choice": PydanticOutputParser(pydantic_object=MultipleChoiceSchema),
        "designation": PydanticOutputParser(pydantic_object=DesignationSchema),
        "expression": PydanticOutputParser(pydantic_object=ExpressionSchema),
        "go_annotation": PydanticOutputParser(pydantic_object=GOAnnotationSchema),
        "summary": PydanticOutputParser(pydantic_object=SummarySchema),
    }
    if question_type not in mapping:
        return PydanticOutputParser(pydantic_object=SummarySchema)
    return mapping[question_type]