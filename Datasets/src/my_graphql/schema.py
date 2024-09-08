import graphene
from my_graphql.utils.file_handler import load_partition_data, load_evaluation_results, calculate_cer_statistics
from my_graphql.types import PartitionData, Statistics


# Define the Query class for fetching HDF5 data and evaluation results
class Query(graphene.ObjectType):
    partition_data = graphene.List(
        PartitionData,  # Expect a list of PartitionData (one for each partition)
        partition=graphene.List(graphene.String, required=True),  # List of partitions (train sizes)
        name_dataset=graphene.String(required=True),
        number_of_rows=graphene.Int(default_value=10)
    )

    def resolve_partition_data(self, info, partition, name_dataset, number_of_rows):
        partition_results = []  # Store results for each partition

        for part in partition:  # Loop through the list of partitions
            # Fetch partition data and evaluation results for each partition
            partition_data, partition_global_total, partition_full_path, total_count = load_partition_data(
                name_dataset, part, number_of_rows
            )
            eval_results = load_evaluation_results(name_dataset, part)

            # Calculate CER statistics for the current partition
            cer_statistics = calculate_cer_statistics(eval_results)

            # Append data for the current partition to the result list
            partition_results.append(
                PartitionData(
                    total_count=total_count,
                    global_total=partition_global_total,
                    path=partition_full_path,
                    data=partition_data,
                    evaluation_data=eval_results,
                    statistics=cer_statistics
                )
            )

        # Return the list of PartitionData, one for each partition
        return partition_results

# Define the schema
schema = graphene.Schema(query=Query)
