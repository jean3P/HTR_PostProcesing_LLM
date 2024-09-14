import { Component, OnInit } from '@angular/core';
import { StatsService } from '../stats.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-stats',
  standalone: true,
  imports: [CommonModule],  // Add CommonModule here to use ngIf and ngFor
  templateUrl: './stats.component.html',
  styleUrls: ['./stats.component.css']
})

export class StatsComponent implements OnInit {
  statistics: any[] = [];
  selectedEvaluationData: any[] = [];  // Store evaluation data
  filteredEvaluationData: any[] = [];  // Store filtered evaluation data
  selectedHtrModel: string = 'Flor_model';
  selectedLlmName: string = 'mistral';
  selectedMethod: string = 'method_1';
  selectedCell: string = '';
  selectedFilter: string = 'all';  // Default filter value

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
    this.selectedLlmName = event.target.value;
    this.loadStats();
  }

  // Method to handle Name Method change
  onMethodChange(event: any): void {
    this.selectedMethod = event.target.value;
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
  }

  loadStats(): void {
    const partitions = ['train_25', 'train_50', 'train_75', 'train_100'];
    const dictionaries = ['washington', 'bentham', 'whitefield', 'iam', 'empty'];

    this.statistics = [];

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

      dictionaries.forEach(dictName => {
        this.statsService.getStats([partition], 'washington', this.selectedHtrModel, this.selectedLlmName, dictName, this.selectedMethod).subscribe(
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

            // Always assign the OCR CER
            statGroup.averageCerOcr = data.statistics?.averageCerOcr || '-';
          },
          (error: any) => {
            console.error(`Error loading stats for partition ${partition} and dictionary ${dictName}`, error);
          }
        );
      });

      this.statistics.push(statGroup);
    });
  }

  // Method to determine the class for best and worst CER values
  getClass(statGroup: any, dataset: string): string {
    const cerValues = [
      statGroup.washington?.cer,
      statGroup.bentham?.cer,
      statGroup.whitefield?.cer,
      statGroup.iam?.cer,
      statGroup.noTraining?.cer
    ];

    const cerValue = statGroup[dataset]?.cer;

    if (cerValue !== null && cerValue !== '-') {
      const minValue = Math.min(...cerValues.filter(v => v !== '-'));
      const maxValue = Math.max(...cerValues.filter(v => v !== '-'));

      if (cerValue === minValue) {
        return 'green-cell';  // Best CER
      } else if (cerValue === maxValue) {
        return 'red-cell';  // Worst CER
      }
    }

    return ''; // Default class
  }

  loadEvaluationData(dataset: string, partition: string): void {
    this.selectedCell = partition + '-' + dataset;

    this.statsService.getEvaluationData([partition], 'washington', this.selectedHtrModel, this.selectedLlmName, dataset, this.selectedMethod).subscribe(
      (response: any) => {
        this.selectedEvaluationData = response.data.partitionData[0].evaluationData;
        this.applyFilter();  // Apply filter after loading new evaluation data
      },
      (error: any) => {
        console.error(`Error loading evaluation data for ${dataset} and partition ${partition}`, error);
      }
    );
  }
}
