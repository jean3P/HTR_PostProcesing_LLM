import { Component } from '@angular/core';
import { StatsComponent } from './stats/stats.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [StatsComponent], // Import StatsComponent
  templateUrl: './app.component.html',
})
export class AppComponent {}
