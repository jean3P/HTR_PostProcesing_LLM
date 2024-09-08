import graphene


# Define the FileInfo type that corresponds to the data in the HDF5 file and evaluation results
class FileInfo(graphene.ObjectType):
    file_name = graphene.String()
    ground_truth = graphene.String()
    predicted_text = graphene.String()  # Predicted text from evaluation
    cer = graphene.Float()  # Character Error Rate (CER)
    image_data = graphene.List(graphene.Float)  # Image data as floats


# Define a new Statistics type to hold CER aggregation data
class Statistics(graphene.ObjectType):
    average_cer = graphene.Float()  # Average CER
    min_cer = graphene.Float()  # Minimum CER
    max_cer = graphene.Float()  # Maximum CER


# Define the PartitionData type to hold the data and additional metadata like the total count
class PartitionData(graphene.ObjectType):
    total_count = graphene.Int()
    global_total = graphene.Int()
    data = graphene.List(FileInfo)  # Actual partition data
    path = graphene.String()
    evaluation_data = graphene.List(FileInfo)  # Evaluation results
    statistics = graphene.Field(Statistics)  # Add the statistics field


