// src/app/stats/stats.component.ts

import { Component, OnInit } from '@angular/core';
import { StatsService } from '../stats.service';
import { CommonModule } from '@angular/common';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatOptionModule } from '@angular/material/core';
import { saveAs } from 'file-saver';
import { forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';

// ✅ All supported LLM names
type LLMName =
  | 'mistral'
  | 'gpt-3.5-turbo'
  | 'gpt-4o-mini'
  | 'llama-3-8B-I'
  | 'qwen-2.5-vl-72b'
  | 'phi-4'
  | 'internalvl3'
  | 'gemini-2.5-pro'
  | 'gpt-4.1-mini'
  | 'claude-sonnet-4';

// ✅ ≤ 6-char acronyms (feel free to tweak to your taste)
const LLM_ACRONYMS: Record<LLMName, string> = {
  mistral: 'Mis-7B',
  'gpt-3.5-turbo': 'G3.5-T',
  'gpt-4o-mini': 'G4o-M',
  'llama-3-8B-I': 'Lm3-8B',
  'qwen-2.5-vl-72b': 'Q2.5V',
  'phi-4': 'Phi-4',
  'internalvl3': 'Int-V3',
  'gemini-2.5-pro': 'Ge2.5P',
  'gpt-4.1-mini': 'G4.1m',
  'claude-sonnet-4': 'Cl-S4',
};

@Component({
  selector: 'app-stats',
  standalone: true,
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule
  ],
  templateUrl: './stats.component.html',
  styleUrls: ['./stats.component.css']
})
export class StatsComponent implements OnInit {
  statistics: { [method: string]: { [llm in LLMName]?: any[] } } = {};
  selectedEvaluationData: any[] = [];
  filteredEvaluationData: any[] = [];
  selectedHtrModel: string = 'Flor_model';
  selectedLlmNames: LLMName[] = ['mistral'];
  selectedDataset: 'washington' | 'bentham' | 'iam' = 'washington';
  selectedCells: { [key: string]: { [llm in LLMName]?: string } } = {};
  selectedFilter: 'all' | 'llm_greater' | 'llm_lesser' | 'llm_equal' = 'all';
  selectedMethods: string[] = []; // Initialize as empty array
  selectedPartition: string = 'train_25';
  logData: string = '';
  filteredLogData: string = '';
  sortColumn: string = '';
  sortOrder: 'asc' | 'desc' = 'asc';
  isLoading: boolean = false;

  constructor(private statsService: StatsService) {}

  ngOnInit(): void {
    // Don't load stats on init if no methods are selected
    if (this.selectedMethods.length > 0) {
      this.loadStats();
    }
  }

  onModelChange(event: any): void {
    if (this.isLoading) return;
    this.selectedHtrModel = event.target.value;
    if (this.selectedMethods.length > 0) {
      this.loadStats();
    }
  }

  onFilterChange(event: any): void {
    this.selectedFilter = event.target.value;
    this.applyFilter();
  }

  onLlmChange(event: any): void {
    if (this.isLoading) return;
    this.selectedLlmNames = event.value as LLMName[];
    if (this.selectedMethods.length > 0) {
      this.loadStats();
    }
  }

  onDatasetChange(event: any): void {
    if (this.isLoading) return;
    this.selectedDataset = event.target.value;
    if (this.selectedMethods.length > 0) {
      this.loadStats();
    }
  }

  onMethodChange(event: any): void {
    if (this.isLoading) return;
    this.selectedMethods = event.value;
    console.log('Selected methods:', this.selectedMethods); // Debug log
    if (this.selectedMethods.length > 0) {
      this.loadStats();
    }
  }

  sortTable(column: string): void {
    const columnMap: { [key: string]: string } = {
      cerLlm: 'cerLlm',
      cerOcr: 'cerOcr',
      werLlm: 'werLlm',
      werOcr: 'werOcr'
    };

    const actualColumn = columnMap[column] || column;

    if (this.sortColumn === actualColumn) {
      this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortColumn = actualColumn;
      this.sortOrder = 'asc';
    }

    this.filteredEvaluationData.sort((a, b) => {
      const valueA = a[actualColumn];
      const valueB = b[actualColumn];

      if (valueA === null || valueA === undefined) return 1;
      if (valueB === null || valueB === undefined) return -1;

      if (typeof valueA === 'string') {
        return this.sortOrder === 'asc'
          ? valueA.localeCompare(valueB)
          : valueB.localeCompare(valueA);
      } else {
        return this.sortOrder === 'asc' ? valueA - valueB : valueB - valueA;
      }
    });
  }

  clearSelection(): void {
    this.selectedCells = {};
    this.selectedEvaluationData = [];
    this.filteredEvaluationData = [];
    this.logData = '';
    this.filteredLogData = '';
    this.sortColumn = '';
  }

  applyFilter(): void {
    if (!this.selectedEvaluationData.length) {
      this.filteredEvaluationData = [];
      this.filteredLogData = '';
      return;
    }

    switch (this.selectedFilter) {
      case 'llm_greater':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(d => d.cerLlm > d.cerOcr);
        break;
      case 'llm_lesser':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(d => d.cerLlm < d.cerOcr);
        break;
      case 'llm_equal':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(d => d.cerLlm === d.cerOcr);
        break;
      default:
        this.filteredEvaluationData = [...this.selectedEvaluationData];
        break;
    }
    this.filteredLogData = this.getFilteredLogs(this.filteredEvaluationData);
  }

  getFilteredLogs(filteredEvaluationData: any[]): string {
    if (!this.logData) return '';

    const escapeRegex = (text: string): string =>
      text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&');

    const filteredLogs = filteredEvaluationData.map(data => {
      const escapedPredictedText = escapeRegex(data.predictedTextOcr);
      const startProcessingRegex = new RegExp(`Start processing text line: '${escapedPredictedText}'`);
      const finishedProcessingRegex = new RegExp(`Finished processing text line: ${escapedPredictedText} ===>`);

      const startLogMatch = this.logData.match(startProcessingRegex);
      const endLogMatch = this.logData.match(finishedProcessingRegex);

      if (startLogMatch && endLogMatch) {
        const startIndex = this.logData.indexOf(startLogMatch[0]);
        const endIndex = this.logData.indexOf(endLogMatch[0]) + endLogMatch[0].length;
        return this.logData.substring(startIndex, endIndex);
      }
      return '';
    });

    return filteredLogs.filter(Boolean).join('\n');
  }

  clearTable(): void {
    this.filteredEvaluationData = [];
    this.selectedEvaluationData = [];
    this.selectedCells = {};
    this.logData = '';
    this.filteredLogData = '';
  }

  // Display acronyms in headings
  getDisplayLlmName(llm: string): string {
    const key = llm as LLMName;
    return LLM_ACRONYMS[key] ?? llm;
  }

  refreshData(): void {
    if (this.isLoading) return;
    if (this.selectedMethods.length > 0) {
      this.loadStats();
    } else {
      alert('Please select at least one method before refreshing.');
    }
  }

  formatPartitionName(partition: string): string {
    let name = partition.replace('train_', '');
    let isEmpty = false;

    if (name.includes('_empty')) {
      isEmpty = true;
      name = name.replace('_empty', '');
    }

    if (name.includes('_remaining_')) {
      const parts = name.split('_remaining_');
      const trainPercent = parts[0];
      const remainingPercent = parts[1];
      let displayName = `${trainPercent}% + ${remainingPercent}%`;
      if (isEmpty) {
        displayName += '';
      }
      return displayName;
    } else {
      return `${name}%`;
    }
  }

  loadStats(): void {
    if (this.isLoading) return;

    if (this.selectedMethods.length === 0) {
      console.log('No methods selected, skipping stats loading');
      return;
    }

    this.isLoading = true;
    this.statsService.resetRequestCounter();

    const partitions = ['train_25', 'train_50', 'train_75', 'train_100'];
    const dictionaries = ['washington', 'bentham', 'iam', 'empty'];

    this.statistics = {};

    console.log(`Loading stats for ${this.selectedMethods.length} methods and ${this.selectedLlmNames.length} LLMs`);

    this.loadStatsSequentially(partitions, dictionaries)
      .catch(error => {
        console.error('❌ Error loading stats:', error);
        alert('Error loading statistics. Please check the console for details.');
      })
      .finally(() => {
        this.isLoading = false;
        console.log('Stats loading completed. Total methods with data:', Object.keys(this.statistics).length);
      });
  }

  private async loadStatsSequentially(partitions: string[], dictionaries: string[]): Promise<void> {
    for (const method of this.selectedMethods) {
      console.log(`Loading data for method: ${method}`);
      this.statistics[method] = {};

      for (const llmName of this.selectedLlmNames) {
        this.statistics[method][llmName] = [];
        await this.loadStatsForMethodAndLLM(method, llmName, partitions, dictionaries);

        this.statistics[method][llmName] = this.statistics[method][llmName].sort(
          (a, b) => partitions.indexOf(a.partition) - partitions.indexOf(b.partition)
        );
      }
    }
  }

  private async loadStatsForMethodAndLLM(
    method: string,
    llmName: LLMName,
    partitions: string[],
    dictionaries: string[]
  ): Promise<void> {

    for (const partition of partitions) {
      let statGroup = {
        partition: partition,
        averageCerOcr: '-',
        averageWerOcr: '-',
        washington: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'},
        bentham: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'},
        iam: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'},
        noTraining: {cer: '-', wer: '-', reduction: '-', werReduction: '-', averageConfidence: '-'}
      };

      const requests = dictionaries.map(dictName =>
        this.statsService.getStats([partition], this.selectedDataset, this.selectedHtrModel, llmName, dictName, method)
          .pipe(
            catchError(error => {
              console.error(`❌ Error loading stats for ${method}-${partition}-${dictName}:`, error);
              return of({ data: { partitionData: [] }, errors: [{ message: error.message }] });
            })
          )
      );

      try {
        const responses = await forkJoin(requests).toPromise();

        if (!responses) {
          console.error(`❌ No responses received for method ${method}, partition ${partition}`);
          continue;
        }

        responses.forEach((response: any, index: number) => {
          const dictName = dictionaries[index];

          if (response.errors && response.errors.length > 0) {
            console.error(`❌ GraphQL errors for ${method}-${partition}-${dictName}:`, response.errors);
            return;
          }

          if (!response.data?.partitionData || response.data.partitionData.length === 0) {
            console.warn(`⚠️ No partition data returned for ${method}-${partition}-${dictName}`);
            return;
          }

          const data = response.data.partitionData[0];

          if (!data || !data.statistics) {
            console.warn(`⚠️ No statistics available for method ${method}, partition ${partition} and dictionary ${dictName}`);
            return;
          }

          const stats = data.statistics;

          switch (dictName) {
            case 'washington':
              statGroup.washington.cer = stats.averageCerLlm?.toFixed(3) || '-';
              statGroup.washington.wer = stats.averageWerLlm?.toFixed(3) || '-';
              statGroup.washington.reduction = stats.cerReductionPercentage?.toFixed(3) || '-';
              statGroup.washington.werReduction = stats.werReductionPercentage?.toFixed(3) || '-';
              statGroup.washington.averageConfidence = stats.averageConfidence?.toFixed(3) || '-';
              break;
            case 'bentham':
              statGroup.bentham.cer = stats.averageCerLlm?.toFixed(3) || '-';
              statGroup.bentham.wer = stats.averageWerLlm?.toFixed(3) || '-';
              statGroup.bentham.reduction = stats.cerReductionPercentage?.toFixed(3) || '-';
              statGroup.bentham.werReduction = stats.werReductionPercentage?.toFixed(3) || '-';
              statGroup.bentham.averageConfidence = stats.averageConfidence?.toFixed(3) || '-';
              break;
            case 'iam':
              statGroup.iam.cer = stats.averageCerLlm?.toFixed(3) || '-';
              statGroup.iam.wer = stats.averageWerLlm?.toFixed(3) || '-';
              statGroup.iam.reduction = stats.cerReductionPercentage?.toFixed(3) || '-';
              statGroup.iam.werReduction = stats.werReductionPercentage?.toFixed(3) || '-';
              statGroup.iam.averageConfidence = stats.averageConfidence?.toFixed(3) || '-';
              break;
            case 'empty':
              statGroup.noTraining.cer = stats.averageCerLlm?.toFixed(3) || '-';
              statGroup.noTraining.wer = stats.averageWerLlm?.toFixed(3) || '-';
              statGroup.noTraining.reduction = stats.cerReductionPercentage?.toFixed(3) || '-';
              statGroup.noTraining.werReduction = stats.werReductionPercentage?.toFixed(3) || '-';
              statGroup.noTraining.averageConfidence = stats.averageConfidence?.toFixed(3) || '-';
              if (statGroup.averageCerOcr === '-') {
                statGroup.averageCerOcr = stats.averageCerOcr?.toFixed(3) || '-';
                statGroup.averageWerOcr = stats.averageWerOcr?.toFixed(3) || '-';
              }
              break;
          }
        });

        (this.statistics[method][llmName] as any[]).push(statGroup);

      } catch (error) {
        console.error(`❌ Error processing method ${method}, partition ${partition}:`, error);
      }

      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  getClass(statGroup: any, dataset: string, method: string, llm: LLMName): string {
    const selectedCell = this.selectedCells[method]?.[llm];
    let className = '';

    if (selectedCell === `${statGroup.partition}-${dataset}`) className += 'selected-cell ';

    const noTrainingCer = parseFloat(statGroup.noTraining?.cer);
    const datasetCer = parseFloat(statGroup[this.selectedDataset]?.cer);
    const currentCer = parseFloat(statGroup[dataset]?.cer);

    if (!isNaN(noTrainingCer) && !isNaN(datasetCer) && !isNaN(currentCer)) {
      const noTrainingIsBetter = noTrainingCer < datasetCer;
      const datasetIsBetter = datasetCer < noTrainingCer;

      if (dataset === 'noTraining' && noTrainingIsBetter) className += 'green-cell';
      else if (dataset === this.selectedDataset && datasetIsBetter) className += 'green-cell';
      else if (dataset === 'noTraining' && datasetIsBetter) className += 'red-cell';
      else if (dataset === this.selectedDataset && noTrainingIsBetter) className += 'red-cell';
    }

    return className.trim();
  }

  loadEvaluationData(method: string, dataset: string, partition: string, llm: LLMName): void {
    this.selectedPartition = partition;
    this.clearSelection();

    if (!this.selectedCells[method]) this.selectedCells[method] = {};
    this.selectedCells[method][llm] = partition + '-' + dataset;

    this.statsService.getEvaluationData([partition], this.selectedDataset, this.selectedHtrModel, llm, dataset, method)
      .subscribe(
        (response: any) => {
          if (response?.data?.partitionData?.length > 0) {
            const data = response.data.partitionData[0];
            this.selectedEvaluationData = data.evaluationData || [];
            this.logData = data.logs || '';
            this.applyFilter();
          } else {
            console.error('No valid partitionData found in the response:', response);
            this.clearSelection();
          }
        },
        (error: any) => {
          console.error(`Error loading evaluation data for ${dataset} and partition ${partition}`, error);
          this.clearSelection();
        }
      );
  }

  downloadResults(): void {
    const llmCondition = this.selectedFilter === 'all' ? 'all' : this.selectedFilter.split('_')[1];
    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().split('T')[0];
    const formattedTime = currentDate.toTimeString().split(' ')[0].replace(/:/g, '-');

    const fileName = `results_${this.selectedDataset}_${this.selectedHtrModel}_${this.selectedLlmNames.join('-')}_${this.selectedMethods.join('-')}_${this.selectedPartition}_${llmCondition}_${formattedDate}_${formattedTime}.json`;

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

    const jsonContent = JSON.stringify(resultData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    saveAs(blob, fileName);
  }

  // ✅ Now supports MULTIPLE methods; one LaTeX table per method in a single file
  downloadTable(): void {
    if (this.selectedLlmNames.length < 1 || !this.selectedDataset) {
      alert('Please select at least one LLM and one dataset to generate the table.');
      return;
    }
    if (this.selectedMethods.length < 1) {
      alert('Please select at least one method to generate the table.');
      return;
    }

    const partitions = ['train_25', 'train_50', 'train_75', 'train_100'];

    const formatValue = (value: number, isBold: boolean = false): string => {
      return isNaN(value)
        ? '-'
        : isBold
          ? `\\textbf{${value.toFixed(2)}}`
          : value.toFixed(2);
    };

    const tables: string[] = [];

    for (const method of this.selectedMethods) {
      let tableContent = `
\\begin{table}
\\centering
\\setlength{\\tabcolsep}{1.7pt}
\\caption{Results for ${this.capitalizeFirstLetter(this.selectedDataset)} (Method: ${method}).}
\\label{tab:results_${this.selectedDataset}_${method}}
\\begin{tabular}{ccc||cccc|cccc}
\\toprule
\\multicolumn{3}{c||}{\\textbf{Setup}} & \\multicolumn{4}{c|}{\\textbf{No Word Suggestions}} & \\multicolumn{4}{c}{\\textbf{With Word Suggestions}} \\\\
\\cmidrule(lr){1-3} \\cmidrule(lr){4-7} \\cmidrule(lr){8-11}
\\multicolumn{2}{c}{\\textbf{HTR}} & \\textbf{LLMs} & \\textbf{CER} & \\textbf{CER-r} & \\textbf{WER} & \\textbf{WER-r} & \\textbf{CER} & \\textbf{CER-r} & \\textbf{WER} & \\textbf{WER-r} \\\\
\\midrule
`;

      for (const partition of partitions) {
        const trainPct = partition.replace('train_', '');

        // Get OCR baseline values from the first LLM's data
        let ocrCer: string = '-';
        let ocrWer: string = '-';

        // Collect all LLM data for this partition
        const llmRows: any[] = [];

        for (const llm of this.selectedLlmNames) {
          const groups = this.statistics[method]?.[llm] as any[] | undefined;
          const statGroup = groups?.find(g => g.partition === partition);

          if (statGroup) {
            // Get OCR baseline from the first valid statGroup
            if (ocrCer === '-' && statGroup.averageCerOcr !== '-') {
              ocrCer = formatValue(parseFloat(statGroup.averageCerOcr));
            }
            if (ocrWer === '-' && statGroup.averageWerOcr !== '-') {
              ocrWer = formatValue(parseFloat(statGroup.averageWerOcr));
            }

            // Collect LLM data
            const emptyCer = parseFloat(statGroup.noTraining.cer);
            const emptyCerReduction = parseFloat(statGroup.noTraining.reduction);
            const emptyWer = parseFloat(statGroup.noTraining.wer);
            const emptyWerReduction = parseFloat(statGroup.noTraining.werReduction);

            const datasetStats = statGroup[this.selectedDataset];
            const nonEmptyCer = parseFloat(datasetStats.cer);
            const nonEmptyCerReduction = parseFloat(datasetStats.reduction);
            const nonEmptyWer = parseFloat(datasetStats.wer);
            const nonEmptyWerReduction = parseFloat(datasetStats.werReduction);

            llmRows.push({
              llm: LLM_ACRONYMS[llm],
              emptyCer,
              emptyCerReduction,
              emptyWer,
              emptyWerReduction,
              nonEmptyCer,
              nonEmptyCerReduction,
              nonEmptyWer,
              nonEmptyWerReduction
            });
          }
        }

        // Check if this method uses word suggestions
        const usesWordSuggestions = !method.toLowerCase().includes('empty') &&
          !method.toLowerCase().includes('baseline') &&
          !method.toLowerCase().includes('no_dict') &&
          !method.toLowerCase().includes('promptor');

        // Always output at least 3 rows: Train, CER, WER
        const minRows = 3;
        const totalRows = Math.max(llmRows.length, minRows);

        for (let i = 0; i < totalRows; i++) {
          let firstCol = '';
          let secondCol = '';
          let llmName = '';
          let noSuggestionsData = { cer: '-', cerReduction: '-', wer: '-', werReduction: '-' };
          let withSuggestionsData = { cer: '-', cerReduction: '-', wer: '-', werReduction: '-' };

          // First column logic
          if (i === 0) {
            firstCol = 'Train';
            secondCol = `${trainPct}\\%`;
          } else if (i === 1) {
            firstCol = 'CER';
            secondCol = ocrCer;
          } else if (i === 2) {
            firstCol = 'WER';
            secondCol = ocrWer;
          }

          // Process LLM data if available for this row
          if (i < llmRows.length) {
            const row = llmRows[i];
            llmName = row.llm;

            // Determine if we should bold the CER values
            const boldEmpty = !isNaN(row.emptyCer) && !isNaN(row.nonEmptyCer) && row.emptyCer < row.nonEmptyCer;
            const boldNonEmpty = !isNaN(row.emptyCer) && !isNaN(row.nonEmptyCer) && row.nonEmptyCer < row.emptyCer;

            // Format values for "No Word Suggestions"
            noSuggestionsData = {
              cer: isNaN(row.emptyCer) ? '-' : formatValue(row.emptyCer, boldEmpty && usesWordSuggestions),
              cerReduction: isNaN(row.emptyCerReduction) ? '-' : formatValue(row.emptyCerReduction),
              wer: isNaN(row.emptyWer) ? '-' : formatValue(row.emptyWer),
              werReduction: isNaN(row.emptyWerReduction) ? '-' : formatValue(row.emptyWerReduction)
            };

            // Format values for "With Word Suggestions"
            if (usesWordSuggestions) {
              withSuggestionsData = {
                cer: isNaN(row.nonEmptyCer) ? '-' : formatValue(row.nonEmptyCer, boldNonEmpty),
                cerReduction: isNaN(row.nonEmptyCerReduction) ? '-' : formatValue(row.nonEmptyCerReduction),
                wer: isNaN(row.nonEmptyWer) ? '-' : formatValue(row.nonEmptyWer),
                werReduction: isNaN(row.nonEmptyWerReduction) ? '-' : formatValue(row.nonEmptyWerReduction)
              };
            }
          }

          tableContent += `${firstCol} & ${secondCol} & ${llmName}`;
          tableContent += ` & ${noSuggestionsData.cer} & ${noSuggestionsData.cerReduction} & ${noSuggestionsData.wer} & ${noSuggestionsData.werReduction}`;
          tableContent += ` & ${withSuggestionsData.cer} & ${withSuggestionsData.cerReduction} & ${withSuggestionsData.wer} & ${withSuggestionsData.werReduction} \\\\ \n`;
        }

        tableContent += '\\midrule\n';
      }

      tableContent += `
\\end{tabular}
\\end{table}
`;
      tables.push(tableContent);
    }

    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().split('T')[0];
    const formattedTime = currentDate.toTimeString().split(' ')[0].replace(/:/g, '-');
    const fileName = `results_${this.selectedDataset}_tables_${formattedDate}_${formattedTime}.tex`;

    const blob = new Blob([tables.join('\n\n')], { type: 'text/plain' });
    saveAs(blob, fileName);
  }

  canDownloadTable(): boolean {
    return this.selectedLlmNames.length >= 1 && this.selectedMethods.length >= 1 && !!this.selectedDataset;
  }

  capitalizeFirstLetter(text: string): string {
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  testConnection(): void {
    this.statsService.testGraphQLConnection().subscribe(
      response => {
        if (response.error) {
          alert('❌ GraphQL connection failed. Check console for details.');
        } else {
          alert('✅ GraphQL connection successful!');
        }
      },
      error => {
        console.error('Test error:', error);
        alert('❌ Connection test failed. Check console for details.');
      }
    );
  }

  // Debug method to check loaded data
  debugStatistics(): void {
    console.log('=== Debug Statistics ===');
    console.log('Selected Methods:', this.selectedMethods);
    console.log('Selected LLMs:', this.selectedLlmNames);
    console.log('Statistics Object:', this.statistics);

    this.selectedMethods.forEach(method => {
      console.log(`\nData for ${method}:`);
      if (this.statistics[method]) {
        this.selectedLlmNames.forEach(llm => {
          console.log(`  ${llm}:`, this.statistics[method][llm]);
        });
      } else {
        console.log('  NO DATA');
      }
    });
  }

  downloadComparativeTable(): void {
    if (this.selectedLlmNames.length < 1 || !this.selectedDataset) {
      alert('Please select at least one LLM and one dataset to generate the comparative table.');
      return;
    }

    // Filter to only get promptOR methods
    const promptMethods = this.selectedMethods.filter(method => method.startsWith('promptOR_'));
    if (promptMethods.length < 1) {
      alert('Please select at least one promptOR method to generate the comparative table.');
      return;
    }

    const partitions = ['train_25', 'train_50', 'train_75', 'train_100'];

    const formatValue = (value: number, isBold: boolean = false): string => {
      return isNaN(value)
        ? '-'
        : isBold
          ? `\\textbf{${value.toFixed(2)}}`
          : value.toFixed(2);
    };

    // Use the methods in the order they were selected (no sorting)
    const methodsInOrder = promptMethods;

    // Caption formatting - use "through" for ranges
    let captionMethods = '';
    if (methodsInOrder.length === 1) {
      captionMethods = methodsInOrder[0].replace(/_/g, '\\_');
    } else if (methodsInOrder.length === 2) {
      captionMethods = `${methodsInOrder[0].replace(/_/g, '\\_')} and ${methodsInOrder[1].replace(/_/g, '\\_')}`;
    } else {
      // For consecutive promptOR methods, use "through"
      const firstMethod = methodsInOrder[0];
      const lastMethod = methodsInOrder[methodsInOrder.length - 1];
      captionMethods = `${firstMethod.replace(/_/g, '\\_')} through ${lastMethod.replace(/_/g, '\\_')}`;
    }

    let tableContent = `\\begin{table}[htbp]
\\centering
\\caption{Comparative Results for All Methods (${captionMethods}) on ${this.capitalizeFirstLetter(this.selectedDataset)} Dataset}
\\label{tab:results_${this.selectedDataset}_all_methods}
\\resizebox{\\linewidth}{!}{%  % Changed from \\textwidth to \\linewidth
\\begin{tabular}{cc|c||${'cccc|'.repeat(methodsInOrder.length).slice(0, -1)}}
\\toprule
\\multicolumn{3}{c||}{\textbf{HTR Setup}} & ${methodsInOrder.map(method => `\\multicolumn{4}{c${method === methodsInOrder[methodsInOrder.length - 1] ? '' : '|'}}{\\textbf{${method.replace(/_/g, '\\_')}}}`).join(' & ')} \\\\
`;

    // Add cmidrule for each method
    tableContent += `\\cmidrule(lr){1-3} `;
    methodsInOrder.forEach((method, index) => {
      const startCol = 4 + (index * 4);
      const endCol = startCol + 3;
      tableContent += `\\cmidrule(lr){${startCol}-${endCol}} `;
    });
    tableContent += '\n';

    // Add sub-headers
    tableContent += `\\textbf{Train} & \\textbf{Baseline} & \\textbf{LLM} & `;
    const subHeaders = methodsInOrder.map(() => '\\textbf{CER} & \\textbf{CER-r} & \\textbf{WER} & \\textbf{WER-r}').join(' & ');
    tableContent += subHeaders + ' \\\\\n\\midrule\n';

    // Process each partition
    for (let partitionIndex = 0; partitionIndex < partitions.length; partitionIndex++) {
      const partition = partitions[partitionIndex];
      const trainPct = partition.replace('train_', '');

      // Add comment for clarity
      tableContent += `% Train ${trainPct}%\n`;

      // Get baseline values from first available method/LLM
      let baselineCer = '-';
      let baselineWer = '-';

      // Find baseline values
      for (const method of methodsInOrder) {
        for (const llm of this.selectedLlmNames) {
          const groups = this.statistics[method]?.[llm] as any[] | undefined;
          const statGroup = groups?.find(g => g.partition === partition);
          if (statGroup && statGroup.averageCerOcr !== '-') {
            // Convert to number and format with 2 decimal places
            const cerValue = parseFloat(statGroup.averageCerOcr);
            const werValue = parseFloat(statGroup.averageWerOcr);
            baselineCer = isNaN(cerValue) ? '-' : cerValue.toFixed(2);
            baselineWer = isNaN(werValue) ? '-' : werValue.toFixed(2);
            break;
          }
        }
        if (baselineCer !== '-') break;
      }

      // Use the total count of selected LLMs for multirow
      const totalLlmCount = this.selectedLlmNames.length;

      // Process EVERY selected LLM
      this.selectedLlmNames.forEach((llm, llmIndex) => {
        // Prepare the row data first to determine best CER across all methods
        let rowData: { method: string; cer: number; cerReduction: number; wer: number; werReduction: number; hasData: boolean }[] = [];

        methodsInOrder.forEach(method => {
          const groups = this.statistics[method]?.[llm] as any[] | undefined;
          const statGroup = groups?.find(g => g.partition === partition);

          if (statGroup && statGroup.noTraining) {
            const cer = parseFloat(statGroup.noTraining.cer);
            const cerReduction = parseFloat(statGroup.noTraining.reduction);
            const wer = parseFloat(statGroup.noTraining.wer);
            const werReduction = parseFloat(statGroup.noTraining.werReduction);

            rowData.push({
              method,
              cer,
              cerReduction,
              wer,
              werReduction,
              hasData: true
            });
          } else {
            rowData.push({
              method,
              cer: NaN,
              cerReduction: NaN,
              wer: NaN,
              werReduction: NaN,
              hasData: false
            });
          }
        });

        // Find the best (lowest) CER value for this row
        const validCers = rowData.filter(d => d.hasData && !isNaN(d.cer)).map(d => d.cer);
        const bestCer = validCers.length > 0 ? Math.min(...validCers) : NaN;

        // Start building the row
        if (llmIndex === 0) {
          // First row: add multirow for train and baseline
          tableContent += `\\multirow{${totalLlmCount}}{*}{${trainPct}\\%} & `;
          tableContent += `\\multirow{${totalLlmCount}}{*}{\\parbox{2cm}{\\centering CER: ${baselineCer}\\\\WER: ${baselineWer}}} \n`;
        } else {
          // Other rows: two empty cells for the multirow columns
          tableContent += ` & `;
        }

        // Add LLM name
        tableContent += `& ${LLM_ACRONYMS[llm]}`;

        // Add data for each method
        rowData.forEach(data => {
          if (data.hasData) {
            const isBestCer = !isNaN(data.cer) && data.cer === bestCer;
            tableContent += ` & ${formatValue(data.cer, isBestCer)}`;
            tableContent += ` & ${formatValue(data.cerReduction, isBestCer)}`;
            tableContent += ` & ${formatValue(data.wer, isBestCer)}`;
            tableContent += ` & ${formatValue(data.werReduction, isBestCer)}`;
          } else {
            tableContent += ` & - & - & - & -`;
          }
        });

        tableContent += ' \\\\\n';
      });

      // Add midrule between partitions (but not after the last one)
      if (partitionIndex < partitions.length - 1) {
        tableContent += '\\midrule\n';
      }
    }

    tableContent += `\\bottomrule
\\end{tabular}
}
\\end{table}`;

    // Save the file
    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().split('T')[0];
    const formattedTime = currentDate.toTimeString().split(' ')[0].replace(/:/g, '-');
    const fileName = `comparative_${this.selectedDataset}_${this.selectedHtrModel}_${formattedDate}_${formattedTime}.tex`;

    const blob = new Blob([tableContent], { type: 'text/plain' });
    saveAs(blob, fileName);
  }

  canDownloadComparativeTable(): boolean {
    const promptMethods = this.selectedMethods.filter(method => method.startsWith('promptOR_'));
    return this.selectedLlmNames.length >= 1 && promptMethods.length >= 1 && !!this.selectedDataset;
  }
}
