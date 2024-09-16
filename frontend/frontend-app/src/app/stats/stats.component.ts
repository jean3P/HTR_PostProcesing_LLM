import { Component, OnInit } from '@angular/core';
import { StatsService } from '../stats.service';
import { CommonModule } from '@angular/common';
import { MatFormFieldModule } from '@angular/material/form-field';  // Import MatFormFieldModule
import { MatSelectModule } from '@angular/material/select';  // Import MatSelectModule
import { MatOptionModule } from '@angular/material/core';  // Import MatOptionModule

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
  selectedLlmNames: string[] = ['mistral'];
  selectedDataset: string = 'washington'; // Default dataset
  selectedCells: { [key: string]: string } = {};  // Store selected cells per method
  selectedFilter: string = 'all';  // Default filter value
  selectedMethods: string[] = [];  // Store selected methods

  constructor(private statsService: StatsService) {}

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


  // Apply the filter based on the selected option
  applyFilter(): void {
    switch (this.selectedFilter) {
      case 'llm_greater':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(data => data.cerLlm > data.cerOcr);
        break;
      case 'llm_lesser':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(data => data.cerLlm < data.cerOcr);
        break;
      case 'llm_equal':
        this.filteredEvaluationData = this.selectedEvaluationData.filter(data => data.cerLlm === data.cerOcr);
        break;
      default:
        this.filteredEvaluationData = [...this.selectedEvaluationData];  // Show all data
        break;
    }
  }

  // Method to clear the table data
  clearTable(): void {
    this.filteredEvaluationData = [];
    this.selectedEvaluationData = [];
    this.selectedCells = {}; // Reset the selected cell
  }

  refreshData(): void {
    this.loadStats(); // Re-fetch or reload the statistics data
  }

  // Load statistics for each selected method
  loadStats(): void {
    const partitions = ['train_25', 'train_50', 'train_75', 'train_100'];
    const dictionaries = ['washington', 'bentham', 'whitefield', 'iam', 'empty'];

    // Clear existing statistics
    this.statistics = {};

    this.selectedMethods.forEach(method => {
      const methodStats: { [llm: string]: any[] } = {};  // Store method-specific stats per LLM

      this.selectedLlmNames.forEach(llmName => {
        methodStats[llmName] = [];  // Initialize an empty array for each LLM

        partitions.forEach(partition => {
          let statGroup = {
            partition: partition,
            averageCerOcr: null,
            washington: { cer: null, reduction: null },
            bentham: { cer: null, reduction: null },
            whitefield: { cer: null, reduction: null },
            iam: { cer: null, reduction: null },
            noTraining: { cer: null, reduction: null }
          };

          const statPromises = dictionaries.map(dictName => {
            return new Promise<void>((resolve, reject) => {
              this.statsService.getStats([partition], this.selectedDataset, this.selectedHtrModel, llmName, dictName, method).subscribe(
                (response: any) => {
                  const data = response.data.partitionData[0];

                  // Assign values based on dictionary
                  switch (dictName) {
                    case 'washington':
                      statGroup.washington.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.washington.reduction = data.statistics?.cerReductionPercentage || '-';
                      break;
                    case 'bentham':
                      statGroup.bentham.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.bentham.reduction = data.statistics?.cerReductionPercentage || '-';
                      break;
                    case 'whitefield':
                      statGroup.whitefield.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.whitefield.reduction = data.statistics?.cerReductionPercentage || '-';
                      break;
                    case 'iam':
                      statGroup.iam.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.iam.reduction = data.statistics?.cerReductionPercentage || '-';
                      break;
                    case 'empty':
                      statGroup.noTraining.cer = data.statistics?.averageCerLlm || '-';
                      statGroup.noTraining.reduction = data.statistics?.cerReductionPercentage || '-';
                      break;
                  }

                  // Assign OCR CER
                  statGroup.averageCerOcr = data.statistics?.averageCerOcr || '-';
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
          Promise.all(statPromises).then(() => {
            methodStats[llmName].push(statGroup);  // Push the stat group to the correct LLM
          });
        });
      });

      // Store the results for the current method
      this.statistics[method] = methodStats;
    });
  }

  // Class logic to determine the style of each table cell
  getClass(statGroup: any, dataset: string, method: string): string {
    const selectedCell = this.selectedCells[method];  // Get the selected cell for this method
    const cerValues = [
      statGroup.washington?.cer,
      statGroup.bentham?.cer,
      statGroup.whitefield?.cer,
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
  loadEvaluationData(method: string, dataset: string, partition: string): void {
    this.selectedCells[method] = partition + '-' + dataset;

    const promisesForLlm = this.selectedLlmNames.map(llmName => {
      return new Promise<void>((resolve, reject) => {
        this.statsService.getEvaluationData([partition], this.selectedDataset, this.selectedHtrModel, llmName, dataset, method).subscribe(
          (response: any) => {
            this.selectedEvaluationData = response.data.partitionData[0].evaluationData;
            this.applyFilter();  // Apply filter after loading new evaluation data
          },
          (error: any) => {
            console.error(`Error loading evaluation data for ${dataset} and partition ${partition}`, error);
          }
        );
      });
    });
    // Wait for all LLM promises to resolve
    Promise.all(promisesForLlm).then(() => {
      console.log('All evaluation data loaded for selected LLMs.');
    });
  }
}

