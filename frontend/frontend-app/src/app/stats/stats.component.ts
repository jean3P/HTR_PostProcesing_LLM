import { Component, OnInit } from '@angular/core';
import { StatsService } from '../stats.service';
import { CommonModule } from '@angular/common';
import { MatFormFieldModule } from '@angular/material/form-field';  // Import MatFormFieldModule
import { MatSelectModule } from '@angular/material/select';  // Import MatSelectModule
import { MatOptionModule } from '@angular/material/core';  // Import MatOptionModule
import { saveAs } from 'file-saver';

type LLMName = 'mistral' | 'gpt-3.5-turbo' | 'gpt-4o-mini';

@Component({
  selector: 'app-stats',
  standalone: true,
  imports: [
    CommonModule,
    MatFormFieldModule,   // Add MatFormFieldModule to imports
    MatSelectModule,
    MatOptionModule
  ],
  templateUrl: './stats.component.html',
  styleUrls: ['./stats.component.css']
})

export class StatsComponent implements OnInit {
  statistics: { [method: string]: { [llm: string]: any[] } } = {};  // Store statistics per method and LLM
  selectedEvaluationData: any[] = [];  // Store evaluation data
  filteredEvaluationData: any[] = [];  // Store filtered evaluation data
  selectedHtrModel: string = 'Flor_model';
  // selectedLlmName: string = 'mistral';
  selectedLlmNames: LLMName[] = ['mistral'];
  selectedDataset: string = 'washington'; // Default dataset
  selectedCells: { [key: string]: { [llm: string]: string } } = {};  // Store selected cells per method and LLM
  selectedFilter: string = 'all';  // Default filter value
  selectedMethods: string[] = [];  // Store selected methods
  selectedPartition: string = 'train_25';
  logData: string =''
  filteredLogData: string = '';

  constructor(private statsService: StatsService) {
  }

  ngOnInit(): void {
    this.loadStats();
  }

  // Method to handle HTR Model change
  onModelChange(event: any): void {
    this.selectedHtrModel = event.target.value;
    this.loadStats();
  }

  // Handle filter change
  onFilterChange(event: any): void {
    this.selectedFilter = event.target.value;
    this.applyFilter();  // Apply the filter when selection changes
  }

  // Method to handle LLM Model change
  onLlmChange(event: any): void {
    this.selectedLlmNames = event.value;
    // Reload statistics when LLM selection changes
    this.loadStats();
  }

  // Method to handle Dataset change
  onDatasetChange(event: any): void {
    this.selectedDataset = event.target.value; // Update selected dataset
    this.loadStats();
  }

  // Method to handle Name Method change
  onMethodChange(event: any): void {
    this.selectedMethods = event.value;

    // Reload statistics when method selection changes
    this.loadStats();
  }

  // Clear all previously selected data
  clearSelection(): void {
    this.selectedCells = {}; // Reset selected cells
    this.selectedEvaluationData = []; // Clear evaluation data
    this.filteredEvaluationData = []; // Clear filtered data
    this.logData = ''; // Clear logs
    this.filteredLogData = ''; // Clear filtered logs
  }


  // Apply the filter based on the selected option
  applyFilter(): void {
    if (!this.selectedEvaluationData.length) {
      // If no data is selected, don't apply the filter
      this.filteredEvaluationData = [];
      this.filteredLogData = '';
      return;
    }

    // Filter evaluation data based on selected criteria
    switch (this.selectedFilter) {
      case 'llm_greater':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(data => data.cerLlm > data.cerOcr);
        this.filteredLogData = this.getFilteredLogs(this.filteredEvaluationData);
        break;
      case 'llm_lesser':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(data => data.cerLlm < data.cerOcr);
        this.filteredLogData = this.getFilteredLogs(this.filteredEvaluationData);
        break;
      case 'llm_equal':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(data => data.cerLlm === data.cerOcr);
        this.filteredLogData = this.getFilteredLogs(this.filteredEvaluationData);
        break;
      default:
        this.filteredEvaluationData = [...this.selectedEvaluationData];  // Show all data
        this.filteredLogData = this.logData;  // Show all logs
        break;
    }
  }

  getFilteredLogs(filteredEvaluationData: any[]): string {
    if (!this.logData) return '';

    // Function to escape special characters for regex
    const escapeRegex = (text: string): string => {
      return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&');
    };

    const filteredLogs = filteredEvaluationData.map(data => {
      const escapedPredictedText = escapeRegex(data.predictedTextOcr);  // Escape special characters
      const startProcessingRegex = new RegExp(`Start processing text line: '${escapedPredictedText}'`);
      const finishedProcessingRegex = new RegExp(`Finished processing text line: ${escapedPredictedText} ===>`);

      // Extract relevant log entries based on start and end processing markers
      const startLogMatch = this.logData.match(startProcessingRegex);
      const endLogMatch = this.logData.match(finishedProcessingRegex);

      if (startLogMatch && endLogMatch) {
        const startIndex = this.logData.indexOf(startLogMatch[0]);
        const endIndex = this.logData.indexOf(endLogMatch[0]) + endLogMatch[0].length;
        return this.logData.substring(startIndex, endIndex);  // Extract log section
      }
      return '';  // Return empty string if no match found
    });

    // Join all filtered logs into a single string with line breaks
    return filteredLogs.filter(log => log).join('\n');
  }




  // Method to clear the table data
  clearTable(): void {
    this.filteredEvaluationData = [];
    this.selectedEvaluationData = [];
    this.selectedCells = {}; // Reset the selected cell
    this.logData = ''
    this.clearSelection()
    this.filteredLogData = ''
  }

  getDisplayLlmName(llm: string): string {
    console.log(llm)
    return llm === 'gpt-3.5' ? 'Gpt-3.5 Turbo' : llm;
  }

  refreshData(): void {
    this.loadStats(); // Re-fetch or reload the statistics data
  }

  // Load statistics for each selected method
  loadStats(): void {
    const partitions = ['train_25', 'train_50', 'train_75', 'train_100'];
    const dictionaries = ['washington', 'bentham', 'iam', 'empty'];

    // Clear existing statistics
    this.statistics = {};

    this.selectedMethods.forEach(method => {
      const methodStats: { [llm: string]: any[] } = {};  // Store method-specific stats per LLM

      this.selectedLlmNames.forEach((llmName: LLMName) => {
        methodStats[llmName] = [];  // Initialize an empty array for each LLM

        partitions.forEach(partition => {
          let statGroup = {
            partition: partition,
            averageCerOcr: '-',
            averageWerOcr: '-',
            washington: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'},
            bentham: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'},
            iam: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'},
            noTraining: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'}
          };

          const statPromises = dictionaries.map(dictName => {
            return new Promise<void>((resolve, reject) => {
              this.statsService.getStats([partition], this.selectedDataset, this.selectedHtrModel, llmName, dictName, method).subscribe(
                (response: any) => {
                  console.log('Full Response loadStats:', response);
                  const data = response.data?.partitionData?.[0];  // Safely access partitionData
                  if (!data) {
                    // If partitionData is null or undefined, set all stats to '-'
                    console.warn(`No partitionData found for partition ${partition} and dictionary ${dictName}`);
                    resolve();
                    return;
                  }
                  // Assign values based on dictionary
                  switch (dictName) {
                    case 'washington':
                      statGroup.washington.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.washington.wer = data.statistics?.averageWerLlm || '-';
                      statGroup.washington.reduction = data.statistics?.cerReductionPercentage || '-';
                      statGroup.washington.werReduction = data.statistics?.werReductionPercentage || '-';
                      statGroup.washington.averageConfidence = data.statistics?.averageConfidence || '-';
                      break;
                    case 'bentham':
                      statGroup.bentham.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.bentham.wer = data.statistics?.averageWerLlm || '-';
                      statGroup.bentham.reduction = data.statistics?.cerReductionPercentage || '-';
                      statGroup.bentham.werReduction = data.statistics?.werReductionPercentage || '-';
                      statGroup.bentham.averageConfidence = data.statistics?.averageConfidence || '-';
                      break;
                    case 'iam':
                      statGroup.iam.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.iam.wer = data.statistics?.averageWerLlm || '-';
                      statGroup.iam.reduction = data.statistics?.cerReductionPercentage || '-';
                      statGroup.iam.werReduction = data.statistics?.werReductionPercentage || '-';
                      statGroup.iam.averageConfidence = data.statistics?.averageConfidence || '-';
                      break;
                    case 'empty':
                      statGroup.noTraining.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.noTraining.wer = data.statistics?.averageWerLlm || '-';
                      statGroup.noTraining.reduction = data.statistics?.cerReductionPercentage || '-';
                      statGroup.noTraining.werReduction = data.statistics?.werReductionPercentage || '-';
                      statGroup.noTraining.averageConfidence = data.statistics?.averageConfidence || '-';
                      break;
                  }

                  // Assign OCR CER
                  statGroup.averageCerOcr = data.statistics?.averageCerOcr || '-';
                  statGroup.averageWerOcr = data.statistics?.averageWerOcr || '-';
                  resolve();
                },
                (error: any) => {
                  console.error(`Error loading stats for partition ${partition} and dictionary ${dictName}`, error);
                  reject();
                }
              );
            });
          });

          // Wait for all promises to resolve before pushing to the LLM-specific stats
          Promise.allSettled(statPromises).then(() => {
            methodStats[llmName].push(statGroup);  // Push the stat group to the correct LLM
            // Sort the statistics to maintain the order of partitions
            methodStats[llmName] = methodStats[llmName].sort(
              (a, b) => partitions.indexOf(a.partition) - partitions.indexOf(b.partition)
            );
          });
        });
      });

      // Store the results for the current method
      this.statistics[method] = methodStats;
    });
  }

  // Class logic to determine the style of each table cell
  getClass(statGroup: any, dataset: string, method: string, llm:string): string {
    const selectedCell = this.selectedCells[method]?.[llm];   // Get the selected cell for this method
    const cerValues = [
      statGroup.washington?.cer,
      statGroup.bentham?.cer,
      statGroup.iam?.cer,
      statGroup.noTraining?.cer
    ];

    const cerValue = statGroup[dataset]?.cer;

    // Check if the current cell is the selected one for this method
    let className = '';
    if (selectedCell === `${statGroup.partition}-${dataset}`) {
      className += 'selected-cell ';  // Highlight selected cell
    }

    // Apply best/worst CER logic
    if (cerValue !== null && cerValue !== '-') {
      const minValue = Math.min(...cerValues.filter(v => v !== '-' && v !== null));
      const maxValue = Math.max(...cerValues.filter(v => v !== '-' && v !== null));

      if (cerValue === minValue) {
        className += 'green-cell';  // Best CER
      } else if (cerValue === maxValue) {
        className += 'red-cell';  // Worst CER
      }
    }

    return className.trim();  // Return the class name, trim to avoid extra spaces
  }

  // Load evaluation data for a specific method, dataset, and partition
  loadEvaluationData(method: string, dataset: string, partition: string, llm: string): void {
    this.selectedPartition = partition;

    // Clear all selections and reset data before loading new data
    this.clearSelection();  // Clear previous selections, data, and logs

    // Initialize the selectedCells object for the current method if not present
    if (!this.selectedCells[method]) {
      this.selectedCells[method] = {};
    }

    // Store the selection for the specific method and LLM
    this.selectedCells[method][llm] = partition + '-' + dataset;

    // Load evaluation data for the selected partition, dataset, method, and LLM
    this.statsService.getEvaluationData([partition], this.selectedDataset, this.selectedHtrModel, llm, dataset, method).subscribe(
      (response: any) => {
        console.log('Full Response loadEvaluationData:', response);

        if (response && response.data && response.data.partitionData && response.data.partitionData.length > 0) {
          const data = response.data.partitionData[0];
          this.selectedEvaluationData = data.evaluationData || [];
          this.logData = data.logs || '';  // Capture logs data
          this.applyFilter();  // Apply filter after loading new evaluation data
        } else {
          // If no valid partition data is found, clear the selection
          console.error('No valid partitionData found in the response:', response);
          this.clearSelection();  // Clear the current selection as no data was found
        }
      },
      (error: any) => {
        console.error(`Error loading evaluation data for ${dataset} and partition ${partition}`, error);
        this.clearSelection();  // Clear the current selection in case of error
      }
    );
  }


// Method to download filtered results as a JSON file
  downloadResults(): void {
    // Format the name of the file
    const llmCondition = this.selectedFilter === 'all' ? 'all' : this.selectedFilter.split('_')[1]; // e.g., 'greater', 'less', 'equal'
    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().split('T')[0]; // YYYY-MM-DD
    const formattedTime = currentDate.toTimeString().split(' ')[0].replace(/:/g, '-'); // HH-MM-SS

    const fileName = `results_${this.selectedDataset}_${this.selectedHtrModel}_${this.selectedLlmNames.join('-')}_${this.selectedMethods.join('-')}_${this.selectedPartition}_${llmCondition}_${formattedDate}_${formattedTime}.json`;

    // Create the data structure as specified in the example
    const resultData = this.filteredEvaluationData.map(item => ({
      file_name: item.fileName,
      ground_truth_label: item.groundTruth,
      OCR: {
        predicted_label: item.predictedTextOcr,
        cer: item.cerOcr,
        wer: item.werOcr
      },
      "Prompt correcting": {
        predicted_label: item.predictedTextLlm,
        cer: item.cerLlm,
        wer: item.werLlm,
        confidence: item.confidence,
        justification: item.justification
      }
    }));

    // Convert to JSON string
    const jsonContent = JSON.stringify(resultData, null, 2); // Pretty-print with indentation

    // Create a Blob and trigger download
    const blob = new Blob([jsonContent], { type: 'application/json' });
    saveAs(blob, fileName); // Use file-saver to download the file
  }

  downloadTable(): void {
    // Check if at least one LLM is selected and one dataset is selected
    if (this.selectedLlmNames.length >= 1 && this.selectedDataset) {
      // Proceed only if one method is selected
      if (this.selectedMethods.length === 1) {
        const method = this.selectedMethods[0];

        // Prepare data for the LaTeX table
        const partitions = ['train_25', 'train_50', 'train_75', 'train_100'];
        const llmDisplayNames: { [key in LLMName]: string } = {
          'mistral': 'Mis-7B',
          'gpt-3.5-turbo': 'G3.5-T',
          'gpt-4o-mini': 'G4o-M'
        };

        // Initialize table content
        let tableContent = `
\\begin{table}[h]
\\centering
\\setlength{\\tabcolsep}{0.4pt}
\\caption{Results for ${this.capitalizeFirstLetter(this.selectedDataset)} Dataset.}
\\label{tab:results_${this.selectedDataset}}
\\begin{tabular}{lccc|cccc|cccc}
    \\toprule
    \\multirow{2}{*}{\\textbf{Size}} & \\multicolumn{2}{c}{\\textbf{OCR}} & \\multirow{2}{*}{\\textbf{LLMs}} & \\multicolumn{4}{c}{\\textbf{Empty Suggestions}} & \\multicolumn{4}{c}{\\textbf{Non-Empty Suggestions}} \\\\ \\cmidrule(lr){2-3} \\cmidrule(lr){5-8} \\cmidrule(lr){9-12}
    & \\textbf{CER} & \\textbf{WER} & & \\textbf{CER} & \\textbf{CER-r} & \\textbf{WER} & \\textbf{WER-r} & \\textbf{CER} & \\textbf{CER-r} & \\textbf{WER} & \\textbf{WER-r} \\\\ \\midrule
  `;

        partitions.forEach((partition) => {
          let firstLlmInPartition = true;
          const numLlms = this.selectedLlmNames.length;

          this.selectedLlmNames.forEach((llm, index) => {
            const statGroup = this.statistics[method]?.[llm]?.find(group => group.partition === partition);

            if (statGroup) {
              // Extract OCR values
              const ocrCer = parseFloat(statGroup.averageCerOcr);
              const ocrWer = parseFloat(statGroup.averageWerOcr);

              // Extract Empty Suggestions (No Training) values
              const emptyCer = parseFloat(statGroup.noTraining.cer);
              const emptyCerReduction = parseFloat(statGroup.noTraining.reduction);
              const emptyWer = parseFloat(statGroup.noTraining.wer);
              const emptyWerReduction = parseFloat(statGroup.noTraining.werReduction);

              // Extract Non-Empty Suggestions (Selected Dataset) values
              const datasetStats = statGroup[this.selectedDataset];
              const nonEmptyCer = parseFloat(datasetStats.cer);
              const nonEmptyCerReduction = parseFloat(datasetStats.reduction);
              const nonEmptyWer = parseFloat(datasetStats.wer);
              const nonEmptyWerReduction = parseFloat(datasetStats.werReduction);

              // Determine best CER values among Empty and Non-Empty Suggestions
              const cerValues = [emptyCer, nonEmptyCer].filter(v => !isNaN(v));
              const bestCer = Math.min(...cerValues);

              // Function to format and bold best CER values
              const formatCerValue = (value: number, bestCer: number): string => {
                const formattedValue = value.toFixed(2);
                return value === bestCer ? `\\textbf{${formattedValue}}` : formattedValue;
              };

              // Function to format WER values (no bold)
              const formatWerValue = (value: number): string => {
                return value.toFixed(2);
              };

              // Build the table row
              let row = '';
              if (firstLlmInPartition) {
                row += `\\multirow{${numLlms}}{*}{\\textbf{${partition.replace('train_', '')}\\%}} & `;
                row += `\\multirow{${numLlms}}{*}{${!isNaN(ocrCer) ? ocrCer.toFixed(2) : '-'}} & `;
                row += `\\multirow{${numLlms}}{*}{${!isNaN(ocrWer) ? ocrWer.toFixed(2) : '-'}} & `;
                firstLlmInPartition = false;
              } else {
                // For subsequent LLMs, skip the 'Size' and 'OCR' columns
                row += ' & & & ';
              }

              row += `${llmDisplayNames[llm]} & `;
              row += !isNaN(emptyCer) ? formatCerValue(emptyCer, bestCer) : '-';
              row += ' & ';
              row += !isNaN(emptyCerReduction) ? emptyCerReduction.toFixed(2) : '-';
              row += ' & ';
              row += !isNaN(emptyWer) ? formatWerValue(emptyWer) : '-';
              row += ' & ';
              row += !isNaN(emptyWerReduction) ? emptyWerReduction.toFixed(2) : '-';
              row += ' & ';
              row += !isNaN(nonEmptyCer) ? formatCerValue(nonEmptyCer, bestCer) : '-';
              row += ' & ';
              row += !isNaN(nonEmptyCerReduction) ? nonEmptyCerReduction.toFixed(2) : '-';
              row += ' & ';
              row += !isNaN(nonEmptyWer) ? formatWerValue(nonEmptyWer) : '-';
              row += ' & ';
              row += !isNaN(nonEmptyWerReduction) ? nonEmptyWerReduction.toFixed(2) : '-';
              row += ' \\\\ ';

              // Add midrule after the last LLM in the partition
              if (index === numLlms - 1) {
                row += '\\midrule\n';
              }

              tableContent += row;
            }
          });
        });

        // Close the LaTeX table
        tableContent += `
\\end{tabular}
\\end{table}
      `;

        // Get current date and time
        const currentDate = new Date();
        const formattedDate = currentDate.toISOString().split('T')[0]; // YYYY-MM-DD
        const formattedTime = currentDate.toTimeString().split(' ')[0].replace(/:/g, '-'); // HH-MM-SS

        // Create the file name with date and time
        const fileName = `results_${this.selectedDataset}_table_${formattedDate}_${formattedTime}.txt`;

        // Create a Blob and trigger download as a .txt file
        const blob = new Blob([tableContent], { type: 'text/plain' });
        saveAs(blob, fileName);
      } else {
        // Show an alert if conditions are not met
        alert('Please select exactly one method to generate the table.');
      }
    } else {
      // Show an alert if conditions are not met
      alert('Please select at least one LLM and one dataset to generate the table.');
    }
  }


  canDownloadTable(): boolean {
    return this.selectedLlmNames.length >= 1 && this.selectedMethods.length === 1 && !!this.selectedDataset;
  }


  capitalizeFirstLetter(text: string): string {
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

}



const llmDisplayNames: { [key in LLMName]: string } = {
  'mistral': 'Mis-7B',
  'gpt-3.5-turbo': 'G3.5-T',
  'gpt-4o-mini': 'G4o-M'
};
