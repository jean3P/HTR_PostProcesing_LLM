// src/app/app.component.ts

import { Component } from '@angular/core';
import { StatsComponent } from './stats/stats.component';
import {MatTab, MatTabContent, MatTabGroup} from "@angular/material/tabs";
import {VisionTestComponent} from "./vision-test/vision-test.component";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [StatsComponent, MatTab, VisionTestComponent, MatTabGroup, MatTabContent],  // Import StatsComponent
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  title = 'frontend-app';
}
