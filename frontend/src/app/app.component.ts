import { Component, OnInit } from '@angular/core';
import { map, Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'teslagofast';
  imageToShow: string | ArrayBuffer | null = '';
  speedImage: string | ArrayBuffer | null = '';
  maxSpeedImage: string | ArrayBuffer | null = '';

  current_speed_x = 0;
  current_speed_y = 0;

  max_speed_x = 0;
  max_speed_y = 0;

  isRecording = false;

  constructor(private http: HttpClient) {

  }

  ngOnInit(): void {
    this.reloadImage();
    this.reloadSpeedImage();
    this.reloadMaxSpeedImage();
    this.http
    .get(`${environment.api}/coords`)
    .subscribe(data => {
      this.current_speed_x = (data as any)[0];
      this.current_speed_y = (data as any)[1];
      this.max_speed_x = (data as any)[2];
      this.max_speed_y = (data as any)[3];
    });

    this.http.get<boolean>(`${environment.api}/is-recording`).subscribe(isRecording => (this.isRecording = isRecording));
  }

  takeSnapshot(): void {
    this.http
    .get(`${environment.api}/save`, { responseType: 'blob' })
    .subscribe(image => {
      let reader = new FileReader(); 
      reader.addEventListener("load", () => {
          this.imageToShow = reader.result; 
      }, false);
  
      if (image) {
          reader.readAsDataURL(image);
      }
    });
  }

  startRecording(): void {
    this.http.put<boolean>(`${environment.api}/start-recording`, {}).subscribe(isRecording => {
      this.isRecording = isRecording;
    })
  }

  stopRecording(): void {
    this.http.put<boolean>(`${environment.api}/stop-recording`, {}).subscribe(isRecording => {
      this.isRecording = isRecording;
    })
  }

  reloadImage(): void {
    this.http
        .get(`${environment.api}/cam`, { responseType: 'blob' })
        .subscribe(image => {
          let reader = new FileReader(); 
          reader.addEventListener("load", () => {
              this.imageToShow = reader.result; 
          }, false);
      
          if (image) {
              reader.readAsDataURL(image);
          }
        });
  }

  reloadSpeedImage(): void {
    this.http
        .get(`${environment.api}/speed-image`, { responseType: 'blob' })
        .subscribe(image => {
          let reader = new FileReader(); 
          reader.addEventListener("load", () => {
              this.maxSpeedImage = reader.result; 
          }, false);
      
          if (image) {
              reader.readAsDataURL(image);
          }
        });
  }

  reloadMaxSpeedImage(): void {
    this.http
        .get(`${environment.api}/max-speed-image`, { responseType: 'blob' })
        .subscribe(image => {
          let reader = new FileReader(); 
          reader.addEventListener("load", () => {
              this.speedImage = reader.result; 
          }, false);
      
          if (image) {
              reader.readAsDataURL(image);
          }
        });
  }


  save(): void {
    this.http.post(`${environment.api}/save`, {current_x: this.current_speed_x, current_y: this.current_speed_y, max_x: this.max_speed_x, max_y: this.max_speed_y}).subscribe(res => {
      this.reloadImage();
    })
  }
}
