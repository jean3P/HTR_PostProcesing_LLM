// src/app/stats/vision-test.component.ts

import { Component, OnInit } from '@angular/core';
import { StatsService } from '../stats.service';
import { CommonModule } from '@angular/common';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatOptionModule } from '@angular/material/core';
import { MatTabsModule } from '@angular/material/tabs';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { saveAs } from 'file-saver';
import { HttpClient } from '@angular/common/http';
import { Observable, forkJoin, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

// Vision-capable LLM models
type VisionLLMName =
  | 'qwen-2.5-vl-72b'
  | 'gpt-4o-mini'
  | 'gemini-2.5-pro'
  | 'claude-sonnet-4'
  | 'internalvl3';

// OCR models
type OCRModel = 'Flor_model' | 'TrOCR_model';

interface TestResult {
  fileName: string;
  imageUrl?: string;
  groundTruth: string;
  ocrOutput: string;
  textOnlyCorrection: string;
  visionCorrection: string;
  ocrCer: number;
  ocrWer: number;
  textOnlyCer: number;
  textOnlyWer: number;
  visionCer: number;
  visionWer: number;
  improvement: number; // Percentage improvement of vision over text-only
  processingTime?: number;
}

interface ComparisonStats {
  totalSamples: number;
  avgOcrCer: number;
  avgTextOnlyCer: number;
  avgVisionCer: number;
  avgImprovement: number;
  betterWithVision: number;
  worseWithVision: number;
  samePerformance: number;
}

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
  selectedVisionLLM: VisionLLMName = 'qwen-2.5-vl-72b';
  selectedOCRModel: OCRModel = 'TrOCR_model';
  selectedDataset: 'washington' | 'bentham' | 'iam' = 'washington';
  selectedPartition: string = 'train_25';

  // Test configuration
  sampleSize: number = 20; // Number of samples to test
  selectedMethod: string = 'promptOR_vision'; // New method for vision testing

  // Results
  testResults: TestResult[] = [];
  comparisonStats: ComparisonStats | null = null;

  // UI state
  isLoading: boolean = false;
  currentProgress: number = 0;
  errorMessage: string = '';

  // Vision LLM display names
  visionLLMNames: Record<VisionLLMName, string> = {
    'qwen-2.5-vl-72b': 'Qwen 2.5 Vision-Language 72B',
    'gpt-4o-mini': 'GPT-4 Vision Mini',
    'gemini-2.5-pro': 'Gemini 2.5 Pro Vision',
    'claude-sonnet-4': 'Claude Sonnet 4 Vision',
    'internalvl3': 'InternalVL3'
  };

  constructor(
    private statsService: StatsService,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    // Initialize component
  }

  onVisionLLMChange(event: any): void {
    this.selectedVisionLLM = event.value;
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

  onSampleSizeChange(event: any): void {
    this.sampleSize = parseInt(event.target.value) || 20;
  }

  async runVisionTest(): Promise<void> {
    if (this.isLoading) return;

    this.isLoading = true;
    this.currentProgress = 0;
    this.errorMessage = '';
    this.testResults = [];
    this.comparisonStats = null;

    try {
      // Step 1: Get sample data from the selected dataset/partition
      const samples = await this.getSampleData();

      if (!samples || samples.length === 0) {
        throw new Error('No samples found for the selected configuration');
      }

      // Step 2: Process each sample
      for (let i = 0; i < Math.min(samples.length, this.sampleSize); i++) {
        this.currentProgress = Math.round((i / this.sampleSize) * 100);

        const sample = samples[i];
        const result = await this.processSample(sample);

        if (result) {
          this.testResults.push(result);
        }

        // Small delay to prevent overwhelming the API
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Step 3: Calculate statistics
      this.calculateComparisonStats();

    } catch (error: any) {
      console.error('Vision test error:', error);
      this.errorMessage = error.message || 'An error occurred during testing';
    } finally {
      this.isLoading = false;
      this.currentProgress = 100;
    }
  }

  private async getSampleData(): Promise<any[]> {
    // This would typically call your GraphQL API to get evaluation data
    // For now, returning mock structure
    return new Promise((resolve) => {
      // In real implementation, this would call:
      // this.statsService.getEvaluationData(
      //   [this.selectedPartition],
      //   this.selectedDataset,
      //   this.selectedOCRModel,
      //   'mistral', // temporary, just to get the data
      //   this.selectedDataset,
      //   'method_1'
      // )

      // Mock data for demonstration
      const mockSamples = Array.from({ length: 30 }, (_, i) => ({
        fileName: `sample_${i + 1}.jpg`,
        groundTruth: 'This is the ground truth text',
        predictedTextOcr: 'This is teh ground truth txt',
        imageUrl: `/images/${this.selectedDataset}/sample_${i + 1}.jpg`
      }));

      resolve(mockSamples);
    });
  }

  private async processSample(sample: any): Promise<TestResult | null> {
    try {
      const startTime = Date.now();

      // Step 1: Get text-only correction (existing approach)
      const textOnlyResult = await this.getTextOnlyCorrection(
        sample.predictedTextOcr
      );

      // Step 2: Get vision-based correction (new approach)
      const visionResult = await this.getVisionCorrection(
        sample.imageUrl,
        sample.predictedTextOcr
      );

      const processingTime = Date.now() - startTime;

      // Calculate metrics
      const ocrCer = this.calculateCER(sample.groundTruth, sample.predictedTextOcr);
      const ocrWer = this.calculateWER(sample.groundTruth, sample.predictedTextOcr);
      const textOnlyCer = this.calculateCER(sample.groundTruth, textOnlyResult);
      const textOnlyWer = this.calculateWER(sample.groundTruth, textOnlyResult);
      const visionCer = this.calculateCER(sample.groundTruth, visionResult);
      const visionWer = this.calculateWER(sample.groundTruth, visionResult);

      const improvement = ((textOnlyCer - visionCer) / textOnlyCer) * 100;

      return {
        fileName: sample.fileName,
        imageUrl: sample.imageUrl,
        groundTruth: sample.groundTruth,
        ocrOutput: sample.predictedTextOcr,
        textOnlyCorrection: textOnlyResult,
        visionCorrection: visionResult,
        ocrCer,
        ocrWer,
        textOnlyCer,
        textOnlyWer,
        visionCer,
        visionWer,
        improvement,
        processingTime
      };

    } catch (error) {
      console.error('Error processing sample:', sample.fileName, error);
      return null;
    }
  }

  private async getTextOnlyCorrection(ocrText: string): Promise<string> {
    // This would call your existing LLM API for text-only correction
    // For demonstration, returning a mock correction
    return new Promise((resolve) => {
      setTimeout(() => {
        // Simulate some correction
        resolve(ocrText.replace('teh', 'the').replace('txt', 'text'));
      }, 100);
    });
  }

  private async getVisionCorrection(imageUrl: string, ocrText: string): Promise<string> {
    // This would call your vision LLM API with both image and text
    // The API endpoint would need to support multimodal input

    // Mock implementation for demonstration
    return new Promise((resolve) => {
      setTimeout(() => {
        // Simulate better correction with vision
        resolve('This is the ground truth text');
      }, 200);
    });

    // Real implementation would look like:
    // const formData = new FormData();
    // formData.append('image', imageFile);
    // formData.append('ocrText', ocrText);
    // formData.append('llmModel', this.selectedVisionLLM);
    // formData.append('prompt', this.getVisionPrompt(ocrText));
    //
    // return this.http.post<{correction: string}>(
    //   'http://localhost:5000/vision-correction',
    //   formData
    // ).pipe(
    //   map(response => response.correction)
    // ).toPromise();
  }

  private getVisionPrompt(ocrText: string): string {
    return `You are an expert in handwriting recognition.
    I'm providing you with:
    1. An image of handwritten text
    2. OCR output: "${ocrText}"

    Please analyze the image and correct any errors in the OCR output.
    Consider the visual context, letter shapes, and writing style to provide the most accurate transcription.

    Return only the corrected text, without any explanation.`;
  }

  private calculateCER(reference: string, hypothesis: string): number {
    // Character Error Rate calculation
    const refChars = reference.split('');
    const hypChars = hypothesis.split('');

    // Simple Levenshtein distance for demonstration
    // In production, use a proper implementation
    const distance = this.levenshteinDistance(refChars, hypChars);
    return (distance / refChars.length) * 100;
  }

  private calculateWER(reference: string, hypothesis: string): number {
    // Word Error Rate calculation
    const refWords = reference.split(' ');
    const hypWords = hypothesis.split(' ');

    const distance = this.levenshteinDistance(refWords, hypWords);
    return (distance / refWords.length) * 100;
  }

  private levenshteinDistance(a: string[], b: string[]): number {
    const matrix: number[][] = [];

    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b[i - 1] === a[j - 1]) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1, // substitution
            matrix[i][j - 1] + 1,     // insertion
            matrix[i - 1][j] + 1      // deletion
          );
        }
      }
    }

    return matrix[b.length][a.length];
  }

  private calculateComparisonStats(): void {
    if (this.testResults.length === 0) return;

    const stats: ComparisonStats = {
      totalSamples: this.testResults.length,
      avgOcrCer: 0,
      avgTextOnlyCer: 0,
      avgVisionCer: 0,
      avgImprovement: 0,
      betterWithVision: 0,
      worseWithVision: 0,
      samePerformance: 0
    };

    this.testResults.forEach(result => {
      stats.avgOcrCer += result.ocrCer;
      stats.avgTextOnlyCer += result.textOnlyCer;
      stats.avgVisionCer += result.visionCer;
      stats.avgImprovement += result.improvement;

      if (result.visionCer < result.textOnlyCer) {
        stats.betterWithVision++;
      } else if (result.visionCer > result.textOnlyCer) {
        stats.worseWithVision++;
      } else {
        stats.samePerformance++;
      }
    });

    stats.avgOcrCer /= stats.totalSamples;
    stats.avgTextOnlyCer /= stats.totalSamples;
    stats.avgVisionCer /= stats.totalSamples;
    stats.avgImprovement /= stats.totalSamples;

    this.comparisonStats = stats;
  }

  downloadResults(): void {
    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().split('T')[0];
    const formattedTime = currentDate.toTimeString().split(' ')[0].replace(/:/g, '-');

    const fileName = `vision_test_${this.selectedDataset}_${this.selectedVisionLLM}_${formattedDate}_${formattedTime}.json`;

    const exportData = {
      configuration: {
        visionLLM: this.selectedVisionLLM,
        ocrModel: this.selectedOCRModel,
        dataset: this.selectedDataset,
        partition: this.selectedPartition,
        sampleSize: this.sampleSize,
        timestamp: new Date().toISOString()
      },
      statistics: this.comparisonStats,
      results: this.testResults
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    saveAs(blob, fileName);
  }

  clearResults(): void {
    this.testResults = [];
    this.comparisonStats = null;
    this.errorMessage = '';
  }
}
