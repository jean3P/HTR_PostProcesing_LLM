import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class StatsService {
  private apiUrl = 'http://localhost:5000/graphql';  // Replace with your actual API endpoint

  constructor(private http: HttpClient) {}

  // Function to query stats based on dynamic parameters
  getStats(
    partition: string[],
    nameDataset: string,
    htrModel: string,
    llmName: string,
    dictName: string,
    nameMethod: string
  ): Observable<any> {
    const query = {
      query: `
        query {
          partitionData(
            partition: ${JSON.stringify(partition)},
            nameDataset: "${nameDataset}",
            htrModel: "${htrModel}",
            llmName: "${llmName}",
            dictName: "${dictName}",
            nameMethod: "${nameMethod}"
          ) {
            statistics {
              averageCerOcr
              averageCerLlm
              cerReductionPercentage
            }
          }
        }
      `
    };
    return this.http.post<any>(this.apiUrl, query);
  }

  getEvaluationData(
    partition: string[],
    nameDataset: string,
    htrModel: string,
    llmName: string,
    dictName: string,
    nameMethod: string
  ): Observable<any> {
    const query = {
      query: `
        query {
          partitionData(
            partition: ${JSON.stringify(partition)},
            nameDataset: "${nameDataset}",
            htrModel: "${htrModel}",
            llmName: "${llmName}",
            dictName: "${dictName}",
            nameMethod: "${nameMethod}"
          ) {
            evaluationData {
              fileName
              groundTruth
              predictedTextOcr
              cerOcr
              predictedTextLlm
              cerLlm
            }
          }
        }
      `
    };
    return this.http.post<any>(this.apiUrl, query);
  }
}
