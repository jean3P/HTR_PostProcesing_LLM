// src/app/stats.service.ts

import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, tap, delay } from 'rxjs/operators';

// Define interfaces for better type safety - FIXED to match your GraphQL schema
export interface Statistics {
  averageCerOcr?: number;
  averageWerOcr?: number;
  averageCerLlm?: number;
  averageWerLlm?: number;
  cerReductionPercentage?: number;
  werReductionPercentage?: number;
  averageConfidence?: number;
  minCerOcr?: number;
  maxCerOcr?: number;
  minCerLlm?: number;
  maxCerLlm?: number;
}

export interface EvaluationData {
  fileName: string;
  groundTruth: string;
  predictedTextOcr: string;
  cerOcr: number;
  werOcr: number;
  predictedTextLlm: string;
  confidence: string;
  cerLlm: number;
  werLlm: number;
  justification: string;
}

export interface PartitionData {
  statistics: Statistics | null;
  evaluationData: EvaluationData[];
  logs: string;
}

export interface GraphQLResponse<T> {
  data: {
    partitionData: T[];
  };
  errors?: Array<{
    message: string;
    locations?: Array<{ line: number; column: number }>;
    path?: Array<string | number>;
  }>;
}

@Injectable({
  providedIn: 'root'
})
export class StatsService {
  private apiUrl = 'http://localhost:5000/graphql';
  private requestCount = 0;

  constructor(private http: HttpClient) {}

  private handleError<T>(operation = 'operation', result?: T) {
    return (error: HttpErrorResponse): Observable<T> => {
      console.error(`${operation} failed:`, error);

      // Log the actual error details
      if (error.error) {
        console.error('Error details:', error.error);
      }

      // Return a safe fallback result
      return of(result as T);
    };
  }

  private logRequest(queryType: string, params: any) {
    this.requestCount++;
    console.log(`🔄 Request #${this.requestCount} - ${queryType}:`, {
      partition: params.partition,
      dataset: params.nameDataset,
      htrModel: params.htrModel,
      llmName: params.llmName,
      dictName: params.dictName,
      method: params.nameMethod
    });
  }

  // Function to query stats based on dynamic parameters - FIXED to match your schema
  getStats(
    partition: string[],
    nameDataset: string,
    htrModel: string,
    llmName: string,
    dictName: string,
    nameMethod: string
  ): Observable<GraphQLResponse<PartitionData>> {

    this.logRequest('getStats', { partition, nameDataset, htrModel, llmName, dictName, nameMethod });

    // Clean and validate parameters
    const cleanParams = {
      partition: partition.filter(p => p && p.trim()),
      nameDataset: nameDataset?.trim(),
      htrModel: htrModel?.trim(),
      llmName: llmName?.trim(),
      dictName: dictName?.trim(),
      nameMethod: nameMethod?.trim()
    };

    if (!cleanParams.partition.length || !cleanParams.nameDataset || !cleanParams.htrModel ||
      !cleanParams.llmName || !cleanParams.dictName || !cleanParams.nameMethod) {
      console.error('❌ Invalid parameters for getStats:', cleanParams);
      return of({
        data: { partitionData: [] },
        errors: [{ message: 'Invalid parameters provided' }]
      });
    }

    // FIXED: Use camelCase field names to match your working GraphQL schema
    const query = {
      query: `
        query GetStats($partition: [String!]!, $nameDataset: String!, $htrModel: String!, $llmName: String!, $dictName: String!, $nameMethod: String!) {
          partitionData(
            partition: $partition
            nameDataset: $nameDataset
            htrModel: $htrModel
            llmName: $llmName
            dictName: $dictName
            nameMethod: $nameMethod
          ) {
            statistics {
              averageCerOcr
              averageWerOcr
              averageCerLlm
              averageWerLlm
              cerReductionPercentage
              werReductionPercentage
              averageConfidence
            }
          }
        }
      `,
      variables: cleanParams
    };

    const headers = { 'Content-Type': 'application/json' };

    return this.http.post<GraphQLResponse<PartitionData>>(this.apiUrl, query, { headers })
      .pipe(
        delay(50), // Small delay to prevent overwhelming the server
        tap(response => {
          if (response.errors) {
            console.error('❌ GraphQL errors in getStats:', response.errors);
          } else {
            console.log('✅ getStats successful');
          }
        }),
        catchError(this.handleError<GraphQLResponse<PartitionData>>('getStats', {
          data: { partitionData: [] },
          errors: [{ message: 'Network or server error' }]
        }))
      );
  }

  getEvaluationData(
    partition: string[],
    nameDataset: string,
    htrModel: string,
    llmName: string,
    dictName: string,
    nameMethod: string
  ): Observable<GraphQLResponse<PartitionData>> {

    this.logRequest('getEvaluationData', { partition, nameDataset, htrModel, llmName, dictName, nameMethod });

    // Clean and validate parameters
    const cleanParams = {
      partition: partition.filter(p => p && p.trim()),
      nameDataset: nameDataset?.trim(),
      htrModel: htrModel?.trim(),
      llmName: llmName?.trim(),
      dictName: dictName?.trim(),
      nameMethod: nameMethod?.trim()
    };

    if (!cleanParams.partition.length || !cleanParams.nameDataset || !cleanParams.htrModel ||
      !cleanParams.llmName || !cleanParams.dictName || !cleanParams.nameMethod) {
      console.error('❌ Invalid parameters for getEvaluationData:', cleanParams);
      return of({
        data: { partitionData: [] },
        errors: [{ message: 'Invalid parameters provided' }]
      });
    }

    // FIXED: Use camelCase field names to match your GraphQL schema
    const query = {
      query: `
        query GetEvaluationData($partition: [String!]!, $nameDataset: String!, $htrModel: String!, $llmName: String!, $dictName: String!, $nameMethod: String!) {
          partitionData(
            partition: $partition
            nameDataset: $nameDataset
            htrModel: $htrModel
            llmName: $llmName
            dictName: $dictName
            nameMethod: $nameMethod
          ) {
            evaluationData {
              fileName
              groundTruth
              predictedTextOcr
              cerOcr
              werOcr
              predictedTextLlm
              confidence
              cerLlm
              werLlm
              justification
            }
            logs
          }
        }
      `,
      variables: cleanParams
    };

    const headers = { 'Content-Type': 'application/json' };

    return this.http.post<GraphQLResponse<PartitionData>>(this.apiUrl, query, { headers })
      .pipe(
        delay(50), // Small delay to prevent overwhelming the server
        tap(response => {
          if (response.errors) {
            console.error('❌ GraphQL errors in getEvaluationData:', response.errors);
          } else {
            console.log('✅ getEvaluationData successful');
          }
        }),
        catchError(this.handleError<GraphQLResponse<PartitionData>>('getEvaluationData', {
          data: { partitionData: [] },
          errors: [{ message: 'Network or server error' }]
        }))
      );
  }

  // Test connection method - FIXED to match your schema
  testGraphQLConnection(): Observable<any> {
    console.log('🧪 Testing GraphQL connection...');

    const simpleQuery = {
      query: `
        query TestConnection {
          partitionData(
            partition: ["train_25"]
            nameDataset: "washington"
            htrModel: "Flor_model"
            llmName: "mistral"
            dictName: "washington"
            nameMethod: "method_1_paper"
          ) {
            statistics {
              averageCerOcr
            }
          }
        }
      `
    };

    const headers = { 'Content-Type': 'application/json' };

    return this.http.post(this.apiUrl, simpleQuery, { headers })
      .pipe(
        tap(response => console.log('✅ GraphQL connection test successful:', response)),
        catchError(error => {
          console.error('❌ GraphQL connection test failed:', error);
          if (error.error) {
            console.error('Error details:', error.error);
          }
          return of({ error: 'Connection failed' });
        })
      );
  }

  // Reset request counter
  resetRequestCounter() {
    this.requestCount = 0;
  }
}
