import graphene
import time
from constants import llm_logs_file
from my_graphql.utils.file_handler import load_partition_data, load_evaluation_results, calculate_cer_statistics, \
    retrieve_log_info
from my_graphql.types import PartitionData, Statistics


# Define the Query class for fetching HDF5 data and evaluation results
class Query(graphene.ObjectType):
    partition_data = graphene.List(
        PartitionData,
        partition=graphene.List(graphene.String, required=True),
        name_dataset=graphene.String(required=True),
        number_of_rows=graphene.Int(default_value=10),
        name_method=graphene.String(required=True),
        htr_model=graphene.String(required=True),
        dict_name=graphene.String(required=True),
        training_sizes=graphene.List(graphene.String, default_value=['train_25', 'train_50', 'train_75', 'train_100']),
        training_suggestion=graphene.List(graphene.String,
                                          default_value=['bentham', 'iam', 'washington', 'whitefield', 'empty']),
        llm_name=graphene.String(required=True)
    )

    def resolve_partition_data(self, info, partition, name_dataset, name_method, number_of_rows, training_sizes,
                               training_suggestion, llm_name, htr_model, dict_name):
        partition_results = []

        for part in partition:
            time.sleep(0.09)
            partition_data, partition_global_total, partition_full_path, total_count = load_partition_data(
                name_dataset, part, number_of_rows
            )
            eval_results = load_evaluation_results(name_dataset, name_method, part, htr_model, llm_name, dict_name)

            # Log evaluation results to check if CER values are available
            # print(f"Evaluation results for {name_dataset}, partition {part}: {eval_results}")

            # Calculate CER statistics for the current partition
            cer_statistics = calculate_cer_statistics(eval_results)

            # Automatically calculate counts for different CER conditions
            run_id = eval_results[0].run_id
            logs = retrieve_log_info(llm_logs_file, run_id)
            print(logs)
            cer_llm_greater_count = sum(1 for result in eval_results if result.cer_llm > result.cer_ocr)
            cer_llm_lesser_count = sum(1 for result in eval_results if result.cer_llm < result.cer_ocr)
            cer_llm_equal_count = sum(1 for result in eval_results if result.cer_llm == result.cer_ocr)

            # Create a function to filter evaluation results based on the requested condition
            def get_filtered_results(condition):
                if condition == 'greater':
                    return [result for result in eval_results if result.cer_llm > result.cer_ocr]
                elif condition == 'lesser':
                    return [result for result in eval_results if result.cer_llm < result.cer_ocr]
                elif condition == 'equal':
                    return [result for result in eval_results if result.cer_llm == result.cer_ocr]
                return eval_results

            # Check what field is being queried and return the corresponding evaluation data
            queried_fields = {field.name.value for field in info.field_nodes[0].selection_set.selections}

            if 'cerLlmGreaterCount' in queried_fields:
                filtered_eval_results = get_filtered_results('greater')
            elif 'cerLlmLesserCount' in queried_fields:
                filtered_eval_results = get_filtered_results('lesser')
            elif 'cerLlmEqualCount' in queried_fields:
                filtered_eval_results = get_filtered_results('equal')
            else:
                filtered_eval_results = eval_results

            # Return the full result including counts
            partition_results.append(
                PartitionData(
                    total_count=total_count,
                    global_total=partition_global_total,
                    path=partition_full_path,
                    data=partition_data,
                    evaluation_data=filtered_eval_results,
                    statistics=cer_statistics,
                    training_sizes=training_sizes,
                    training_suggestion=training_suggestion,
                    llm_name=llm_name,
                    cer_llm_greater_count=cer_llm_greater_count,
                    cer_llm_lesser_count=cer_llm_lesser_count,
                    cer_llm_equal_count=cer_llm_equal_count,
                    run_id=run_id,
                    logs=logs,
                )
            )

        return partition_results


# Define the schema
schema = graphene.Schema(query=Query)
