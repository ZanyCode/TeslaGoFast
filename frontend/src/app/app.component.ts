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

  constructor(private http: HttpClient) {

  }

  ngOnInit(): void {
    this.reloadImage();
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
}
