# repos/HTR_PostProcesing_LLM/Datasets/my_graphql/schema.py

import os.path
import graphene
import time
import logging
from constants import llm_logs_path, llm_logs_file
from my_graphql.utils.file_handler import load_evaluation_results, calculate_cer_statistics, retrieve_log_info
from my_graphql.types import PartitionData, Statistics


# Define the Query class for fetching LLM evaluation results only
class Query(graphene.ObjectType):
    partition_data = graphene.List(
        PartitionData,
        partition=graphene.List(graphene.String, required=True),
        name_dataset=graphene.String(required=True),
        name_method=graphene.String(required=True),
        htr_model=graphene.String(required=True),
        dict_name=graphene.String(required=True),
        llm_name=graphene.String(required=True),
        # Optional parameters
        number_of_rows=graphene.Int(default_value=10),
        load_hdf5_data=graphene.Boolean(default_value=False),  # New flag to control HDF5 loading
        training_sizes=graphene.List(graphene.String, default_value=['train_25', 'train_50', 'train_75', 'train_100']),
        training_suggestion=graphene.List(graphene.String, default_value=['bentham', 'iam', 'washington', 'empty']),
    )

    def resolve_partition_data(self, info, partition, name_dataset, name_method, htr_model, llm_name, dict_name,
                               number_of_rows=10, load_hdf5_data=False, training_sizes=None, training_suggestion=None):
        partition_results = []

        logging.info(f"🚀 Starting GraphQL resolve for partitions: {partition}")
        logging.info(
            f"📊 Parameters: dataset={name_dataset}, method={name_method}, htr_model={htr_model}, llm={llm_name}, dict={dict_name}")

        for part in partition:
            logging.info(f"🔍 Processing partition: {part}")

            try:
                # Initialize default values
                partition_data = []
                partition_global_total = 0
                partition_full_path = ""
                total_count = 0

                # Only load HDF5 data if explicitly requested
                if load_hdf5_data:
                    try:
                        from my_graphql.utils.file_handler import load_partition_data
                        partition_data, partition_global_total, partition_full_path, total_count = load_partition_data(
                            name_dataset, part, number_of_rows
                        )
                        logging.info(f"✅ Successfully loaded HDF5 data for {part}")
                    except Exception as e:
                        logging.error(f"❌ Failed to load HDF5 data for {part}: {e}")
                        # Continue without HDF5 data
                        pass

                # Load evaluation results (this is what we actually need)
                eval_results = load_evaluation_results(name_dataset, name_method, part, htr_model, llm_name, dict_name)
                logging.info(f"📈 Loaded {len(eval_results)} evaluation results for {part}")

                if not eval_results:
                    logging.warning(f"⚠️ No evaluation results found for {part}")
                    # Still create a partition result with empty data
                    partition_results.append(
                        PartitionData(
                            total_count=0,
                            global_total=0,
                            path="",
                            data=[],
                            evaluation_data=[],
                            statistics=None,
                            training_sizes=training_sizes or [],
                            training_suggestion=training_suggestion or [],
                            llm_name=llm_name,
                            cer_llm_greater_count=0,
                            cer_llm_lesser_count=0,
                            cer_llm_equal_count=0,
                            run_id="",
                            logs="No evaluation data found",
                        )
                    )
                    continue

                # Calculate CER statistics for the current partition
                cer_statistics = calculate_cer_statistics(eval_results)
                logging.info(f"📊 Calculated statistics for {part}: {cer_statistics}")

                # Get run_id from evaluation results
                run_id = eval_results[0].run_id if eval_results else ""

                # Handle dict_name for log file lookup
                log_dict_name = 'empty' if dict_name == 'noTraining' else dict_name
                log_file = os.path.join(llm_logs_path,
                                        f'workflow_{name_dataset}_{htr_model}_{llm_name}_{name_method}_{part}_{log_dict_name}.log')
                logs = retrieve_log_info(log_file, run_id)

                # Calculate counts for different CER conditions
                cer_llm_greater_count = sum(1 for result in eval_results if result.cer_llm > result.cer_ocr)
                cer_llm_lesser_count = sum(1 for result in eval_results if result.cer_llm < result.cer_ocr)
                cer_llm_equal_count = sum(1 for result in eval_results if result.cer_llm == result.cer_ocr)

                # Check what field is being queried and return the corresponding evaluation data
                queried_fields = {field.name.value for field in info.field_nodes[0].selection_set.selections}

                if 'cerLlmGreaterCount' in queried_fields:
                    filtered_eval_results = [result for result in eval_results if result.cer_llm > result.cer_ocr]
                elif 'cerLlmLesserCount' in queried_fields:
                    filtered_eval_results = [result for result in eval_results if result.cer_llm < result.cer_ocr]
                elif 'cerLlmEqualCount' in queried_fields:
                    filtered_eval_results = [result for result in eval_results if result.cer_llm == result.cer_ocr]
                else:
                    filtered_eval_results = eval_results

                # Create the partition result
                partition_results.append(
                    PartitionData(
                        total_count=total_count,
                        global_total=partition_global_total,
                        path=partition_full_path,
                        data=partition_data,
                        evaluation_data=filtered_eval_results,
                        statistics=cer_statistics,
                        training_sizes=training_sizes or [],
                        training_suggestion=training_suggestion or [],
                        llm_name=llm_name,
                        cer_llm_greater_count=cer_llm_greater_count,
                        cer_llm_lesser_count=cer_llm_lesser_count,
                        cer_llm_equal_count=cer_llm_equal_count,
                        run_id=run_id,
                        logs=logs,
                    )
                )

                logging.info(f"✅ Successfully processed partition {part}")

            except Exception as e:
                logging.error(f"❌ Error processing partition {part}: {e}")
                # Create an error result instead of failing completely
                partition_results.append(
                    PartitionData(
                        total_count=0,
                        global_total=0,
                        path="",
                        data=[],
                        evaluation_data=[],
                        statistics=None,
                        training_sizes=training_sizes or [],
                        training_suggestion=training_suggestion or [],
                        llm_name=llm_name,
                        cer_llm_greater_count=0,
                        cer_llm_lesser_count=0,
                        cer_llm_equal_count=0,
                        run_id="",
                        logs=f"Error: {str(e)}",
                    )
                )

        logging.info(f"🏁 Completed processing {len(partition_results)} partitions")
        return partition_results


# Define the schema
schema = graphene.Schema(query=Query)
