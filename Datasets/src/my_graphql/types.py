import graphene


# Update FileInfo to include LLM-related fields like confidence and justification
class FileInfo(graphene.ObjectType):
    file_name = graphene.String()
    ground_truth = graphene.String()
    predicted_text_ocr = graphene.String()  # Predicted text from OCR model
    cer_ocr = graphene.Float()  # CER from OCR model
    predicted_text_llm = graphene.String()  # Predicted text after LLM correction
    cer_llm = graphene.Float()  # CER after LLM correction
    confidence = graphene.String()  # Confidence from LLM correction
    justification = graphene.String()  # Justification from LLM correction
    wer_ocr = graphene.Float()
    wer_llm = graphene.Float()
    run_id = graphene.String()
    image_data = graphene.List(graphene.Float)  # Image data as floats


# Define a new Statistics type to hold CER aggregation data
class Statistics(graphene.ObjectType):
    average_cer_ocr = graphene.Float()  # Average OCR CER
    min_cer_ocr = graphene.Float()  # Minimum OCR CER
    max_cer_ocr = graphene.Float()  # Maximum OCR CER
    average_cer_llm = graphene.Float()  # Average LLM CER
    average_wer_llm = graphene.Float()
    average_wer_ocr = graphene.Float()
    min_cer_llm = graphene.Float()  # Minimum LLM CER
    max_cer_llm = graphene.Float()  # Maximum LLM CER
    cer_reduction_percentage = graphene.Float()  # CER reduction percentage
    wer_reduction_percentage = graphene.Float()  # CER reduction percentage


# Add new fields to PartitionData to include LLM and training metadata
class PartitionData(graphene.ObjectType):
    total_count = graphene.Int()
    global_total = graphene.Int()
    data = graphene.List(FileInfo)  # Actual partition data
    path = graphene.String()
    evaluation_data = graphene.List(FileInfo)  # Evaluation results
    statistics = graphene.Field(Statistics)  # Add the statistics field
    training_sizes = graphene.List(graphene.String)  # List of training sizes (train_25, train_50, etc.)
    training_suggestion = graphene.List(graphene.String)  # List of training suggestions
    llm_name = graphene.String()  # Name of the LLM used
    cer_llm_greater_count = graphene.Int()  # New field for count of LLM CER greater than OCR CER
    cer_llm_lesser_count = graphene.Int()
    cer_llm_equal_count = graphene.Int()  # New field for count of LLM CER equal to OCR CER
