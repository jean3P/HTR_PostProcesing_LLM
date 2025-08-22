// src/app/stats/vision-test.component.ts

import { Component, OnInit } from '@angular/core';
import { StatsService, EvaluationData, Statistics } from '../stats.service';
import { CommonModule } from '@angular/common';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatOptionModule } from '@angular/material/core';
import { MatTabsModule } from '@angular/material/tabs';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { saveAs } from 'file-saver';

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

type OCRModel = 'Flor_model' | 'TrOCR_model';
type ViewMode = 'grid' | 'list' | 'compare';

@Component({
  selector: 'app-vision-test',
  standalone: true,
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule,
    MatTabsModule,
    MatProgressSpinnerModule,
    MatButtonModule,
    MatIconModule
  ],
  templateUrl: './vision-test.component.html',
  styleUrls: ['./vision-test.component.css']
})
export class VisionTestComponent implements OnInit {
  // Model selections
  selectedLLM: LLMName = 'mistral';
  selectedOCRModel: OCRModel = 'Flor_model';
  selectedDataset: 'washington' | 'bentham' | 'iam' = 'washington';
  selectedPartition: string = 'train_25';
  selectedMethod: string = 'promptOR_1';

  // Which dictionary was used to load the data
  loadedWithDictionary: string = '';

  // Display configuration
  displayCount: number = 20;
  viewMode: ViewMode = 'grid';

  // Data
  evaluationData: EvaluationData[] = [];
  displayedData: EvaluationData[] = [];
  statistics: Statistics | null = null;
  logData: string = '';

  // UI state
  isLoading: boolean = false;
  errorMessage: string = '';

  // Failed images tracking
  failedImages: Set<string> = new Set();
  imageLoadAttempts: Map<string, number> = new Map();

  constructor(private statsService: StatsService) {}

  ngOnInit(): void {
    console.log('VisionTestComponent initialized');
  }

  onLLMChange(event: any): void {
    this.selectedLLM = event.value;
  }

  onOCRModelChange(event: any): void {
    this.selectedOCRModel = event.value;
  }

  onDatasetChange(event: any): void {
    this.selectedDataset = event.value;
  }

  onPartitionChange(event: any): void {
    this.selectedPartition = event.value;
  }

  onMethodChange(event: any): void {
    this.selectedMethod = event.value;
  }

  onDisplayCountChange(event: any): void {
    this.displayCount = parseInt(event.target.value) || 20;
    this.updateDisplayedData();
  }

  changeView(mode: ViewMode): void {
    this.viewMode = mode;
  }

  loadData(): void {
    if (this.isLoading) return;

    this.isLoading = true;
    this.errorMessage = '';
    this.evaluationData = [];
    this.displayedData = [];
    this.statistics = null;
    this.failedImages.clear();
    this.imageLoadAttempts.clear();
    this.loadedWithDictionary = '';

    // Try to load with 'empty' dictionary first (No Word Suggestions)
    this.tryLoadWithDictionary('empty');
  }

  private tryLoadWithDictionary(dictionary: string): void {
    console.log(`Attempting to load evaluation data with dictionary: ${dictionary}`);

    // Load evaluation data exactly like the stats component does
    this.statsService.getEvaluationData(
      [this.selectedPartition],
      this.selectedDataset,
      this.selectedOCRModel,
      this.selectedLLM,
      dictionary,  // This is the key - can be 'empty' or dataset name
      this.selectedMethod
    ).subscribe(
      (response: any) => {
        console.log('Response for dictionary', dictionary, ':', response);

        if (response?.data?.partitionData?.length > 0) {
          const data = response.data.partitionData[0];
          this.evaluationData = data.evaluationData || [];
          this.logData = data.logs || '';

          if (this.evaluationData.length > 0) {
            console.log(`✅ Found ${this.evaluationData.length} evaluation items with dictionary: ${dictionary}`);
            this.loadedWithDictionary = dictionary;

            // Also load statistics for display
            this.loadStatistics(dictionary);

            this.updateDisplayedData();
            this.errorMessage = '';
          } else if (dictionary === 'empty') {
            // No data with empty dictionary, try with dataset dictionary
            console.log('No data with empty dictionary, trying with dataset dictionary');
            this.tryLoadWithDictionary(this.selectedDataset);
            return;
          } else {
            this.errorMessage = 'No evaluation data found for this configuration.';
          }
        } else {
          if (dictionary === 'empty') {
            this.tryLoadWithDictionary(this.selectedDataset);
            return;
          }
          this.errorMessage = 'No data found for the selected configuration.';
        }

        this.isLoading = false;
      },
      (error: any) => {
        console.error(`Error loading evaluation data with dictionary ${dictionary}:`, error);
        if (dictionary === 'empty') {
          this.tryLoadWithDictionary(this.selectedDataset);
        } else {
          this.errorMessage = 'Failed to load evaluation data. Please check the console for details.';
          this.isLoading = false;
        }
      }
    );
  }

  private loadStatistics(dictionary: string): void {
    this.statsService.getStats(
      [this.selectedPartition],
      this.selectedDataset,
      this.selectedOCRModel,
      this.selectedLLM,
      dictionary,
      this.selectedMethod
    ).subscribe(
      (response) => {
        if (response?.data?.partitionData?.length > 0) {
          const data = response.data.partitionData[0];
          this.statistics = data.statistics || null;
        }
      },
      (error) => {
        console.error('Error loading statistics:', error);
      }
    );
  }

  private updateDisplayedData(): void {
    this.displayedData = this.evaluationData.slice(0, this.displayCount);
    console.log(`Displaying ${this.displayedData.length} of ${this.evaluationData.length} items`);
  }

  getImagePath(fileName: string): string {
    if (!fileName) {
      console.error('getImagePath called with empty fileName');
      return '';
    }

    // Clean the filename
    const cleanFileName = fileName.trim();

    // Map dataset names to folder names (capitalizing first letter)
    const datasetFolder = this.selectedDataset.charAt(0).toUpperCase() + this.selectedDataset.slice(1);

    // Extract just the filename without path
    const baseName = cleanFileName.split('/').pop()?.split('\\').pop() || cleanFileName;

    // Check if fileName already has an extension
    let imageFileName = baseName;
    const hasExtension = /\.(png|jpg|jpeg|gif|bmp)$/i.test(baseName);

    if (!hasExtension) {
      // If no extension, assume .png
      imageFileName = baseName + '.png';
    }

    // Construct the path - use absolute path from root
    const imagePath = `/resources/${datasetFolder}/${imageFileName}`;

    return imagePath;
  }

  onImageLoad(event: any, fileName: string): void {
    console.log('✅ Successfully loaded image:', fileName);
    this.failedImages.delete(fileName);
  }

  handleImageError(event: any, fileName: string): void {
    const attempts = this.imageLoadAttempts.get(fileName) || 0;
    this.imageLoadAttempts.set(fileName, attempts + 1);

    if (attempts >= 3) {
      this.failedImages.add(fileName);
      this.showErrorPlaceholder(event, fileName);
      return;
    }

    // Try different extensions
    const extensions = ['.png', '.jpg', '.jpeg'];
    const baseFileName = fileName.replace(/\.[^/.]+$/, '');
    const currentExt = fileName.match(/\.[^/.]+$/)?.[0] || '.png';
    const remainingExts = extensions.filter(ext => ext !== currentExt);

    if (remainingExts.length > 0 && attempts < remainingExts.length) {
      const nextExt = remainingExts[attempts];
      const datasetFolder = this.selectedDataset.charAt(0).toUpperCase() + this.selectedDataset.slice(1);
      const retryPath = `/resources/${datasetFolder}/${baseFileName}${nextExt}`;
      console.log(`Retry attempt ${attempts + 1} for ${fileName} with extension ${nextExt}`);
      event.target.src = retryPath;
    } else {
      this.failedImages.add(fileName);
      this.showErrorPlaceholder(event, fileName);
    }
  }

  private showErrorPlaceholder(event: any, fileName: string): void {
    const displayFileName = fileName.split('/').pop()?.split('\\').pop() || fileName;
    event.target.src = `data:image/svg+xml;base64,${btoa(`
      <svg width="300" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect width="300" height="100" fill="#f5f5f5" stroke="#ddd" stroke-width="2"/>
        <text x="50%" y="30%" text-anchor="middle" fill="#999" font-family="Arial" font-size="14" dy=".3em">Image not found</text>
        <text x="50%" y="50%" text-anchor="middle" fill="#666" font-family="Arial" font-size="11" dy=".3em">${displayFileName}</text>
        <text x="50%" y="70%" text-anchor="middle" fill="#888" font-family="Arial" font-size="10" dy=".3em">${this.selectedDataset}/${displayFileName}</text>
      </svg>
    `)}`;
    event.target.classList.add('error');
  }

  getImprovementText(item: EvaluationData): string {
    if (item.cerOcr === 0) {
      return item.cerLlm === 0 ? 'Perfect match' : 'OCR was perfect';
    }

    const improvement = ((item.cerOcr - item.cerLlm) / item.cerOcr) * 100;

    if (item.cerLlm < item.cerOcr) {
      return `${improvement.toFixed(1)}% improvement`;
    } else if (item.cerLlm > item.cerOcr) {
      return `${Math.abs(improvement).toFixed(1)}% worse`;
    } else {
      return 'No change';
    }
  }

  getImprovementPercentage(item: EvaluationData): string {
    if (item.cerOcr === 0) {
      return item.cerLlm === 0 ? '0%' : '-∞%';
    }
    const improvement = ((item.cerOcr - item.cerLlm) / item.cerOcr) * 100;
    return improvement > 0 ? `+${improvement.toFixed(2)}%` : `${improvement.toFixed(2)}%`;
  }

  getWERImprovement(item: EvaluationData): string {
    if (item.werOcr === 0) {
      return item.werLlm === 0 ? '0%' : '-∞%';
    }
    const improvement = ((item.werOcr - item.werLlm) / item.werOcr) * 100;
    return improvement > 0 ? `+${improvement.toFixed(2)}%` : `${improvement.toFixed(2)}%`;
  }

  downloadResults(): void {
    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().split('T')[0];
    const formattedTime = currentDate.toTimeString().split(' ')[0].replace(/:/g, '-');

    const fileName = `vision_test_${this.selectedDataset}_${this.selectedLLM}_${this.selectedMethod}_${formattedDate}_${formattedTime}.json`;

    const exportData = {
      configuration: {
        llm: this.selectedLLM,
        ocrModel: this.selectedOCRModel,
        dataset: this.selectedDataset,
        partition: this.selectedPartition,
        method: this.selectedMethod,
        dictionary: this.loadedWithDictionary,
        displayCount: this.displayCount,
        totalItems: this.evaluationData.length,
        timestamp: new Date().toISOString()
      },
      statistics: this.statistics,
      failedImages: Array.from(this.failedImages),
      results: this.displayedData.map(item => ({
        fileName: item.fileName,
        imagePath: this.getImagePath(item.fileName),
        groundTruth: item.groundTruth,
        ocrOutput: item.predictedTextOcr,
        llmCorrection: item.predictedTextLlm,
        confidence: item.confidence,
        justification: item.justification,
        metrics: {
          cerOcr: item.cerOcr,
          cerLlm: item.cerLlm,
          werOcr: item.werOcr,
          werLlm: item.werLlm,
          cerImprovement: this.getImprovementPercentage(item),
          werImprovement: this.getWERImprovement(item)
        }
      }))
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    saveAs(blob, fileName);
  }

  clearResults(): void {
    this.evaluationData = [];
    this.displayedData = [];
    this.statistics = null;
    this.errorMessage = '';
    this.logData = '';
    this.failedImages.clear();
    this.imageLoadAttempts.clear();
    this.loadedWithDictionary = '';
  }

  // Debug method to check what configurations have data
  async debugCheckConfigurations(): Promise<void> {
    console.log('=== Checking Available Configurations ===');
    this.isLoading = true;

    const methods = ['promptOR_1', 'promptOR_2', 'promptOR_3', 'promptOR_4'];
    const llms: LLMName[] = ['mistral', 'gpt-3.5-turbo', 'gpt-4o-mini'];
    const dictionaries = ['empty', this.selectedDataset];

    for (const method of methods) {
      for (const llm of llms) {
        for (const dict of dictionaries) {
          try {
            const response = await this.statsService.getEvaluationData(
              [this.selectedPartition],
              this.selectedDataset,
              this.selectedOCRModel,
              llm,
              dict,
              method
            ).toPromise();

            if (response && response.data && response.data.partitionData && response.data.partitionData[0]) {
              const evaluationData = response.data.partitionData[0].evaluationData;
              const hasData = evaluationData && evaluationData.length > 0;
              if (hasData) {
                const count = evaluationData.length;
                console.log(`✅ ${method} + ${llm} + ${dict}: ${count} items`);
              } else {
                console.log(`❌ ${method} + ${llm} + ${dict}: No data`);
              }
            } else {
              console.log(`❌ ${method} + ${llm} + ${dict}: No response data`);
            }
          } catch (error) {
            console.error(`❌ ${method} + ${llm} + ${dict}: Error`, error);
          }
        }
      }
    }

    this.isLoading = false;
    console.log('=== Check Complete ===');
  }
}
