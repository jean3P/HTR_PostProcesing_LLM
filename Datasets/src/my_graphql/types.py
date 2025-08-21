# repos/HTR_PostProcesing_LLM/Datasets/my_graphql/types.py

import graphene


# Update FileInfo to include LLM-related fields like confidence and justification
class FileInfo(graphene.ObjectType):
    file_name = graphene.String()
    ground_truth = graphene.String()
    predicted_text_ocr = graphene.String()
    cer_ocr = graphene.Float()
    predicted_text_llm = graphene.String()
    cer_llm = graphene.Float()
    confidence = graphene.String()
    justification = graphene.String()
    wer_ocr = graphene.Float()
    wer_llm = graphene.Float()
    run_id = graphene.String()
    image_data = graphene.List(graphene.Float)


# Define Statistics type with consistent camelCase field names
class Statistics(graphene.ObjectType):
    # OCR fields
    average_cer_ocr = graphene.Float()
    average_wer_ocr = graphene.Float()
    min_cer_ocr = graphene.Float()
    max_cer_ocr = graphene.Float()

    # LLM fields
    average_cer_llm = graphene.Float()
    average_wer_llm = graphene.Float()
    min_cer_llm = graphene.Float()
    max_cer_llm = graphene.Float()
    average_confidence = graphene.Float()

    # Reduction percentages
    cer_reduction_percentage = graphene.Float()
    wer_reduction_percentage = graphene.Float()


# PartitionData type for LLM-focused results
class PartitionData(graphene.ObjectType):
    # Optional HTR data (only when loadHdf5Data=true)
    total_count = graphene.Int()
    global_total = graphene.Int()
    data = graphene.List(FileInfo)
    path = graphene.String()

    # LLM evaluation data (always available)
    evaluation_data = graphene.List(FileInfo)
    statistics = graphene.Field(Statistics)

    # Metadata
    training_sizes = graphene.List(graphene.String)
    training_suggestion = graphene.List(graphene.String)
    llm_name = graphene.String()

    # CER comparison counts
    cer_llm_greater_count = graphene.Int()
    cer_llm_lesser_count = graphene.Int()
    cer_llm_equal_count = graphene.Int()

    # Logs and run info
    run_id = graphene.String()
    logs = graphene.String()
