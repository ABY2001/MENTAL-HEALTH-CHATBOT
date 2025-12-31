import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, NgModule, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
  imports: [CommonModule,FormsModule],
})


export class LoginComponent implements OnInit {
  password: any;
  email: any;

  constructor( private http: HttpClient,private router: Router) { }

  ngOnInit() {
  }


getLoginCredentials(){
    console.log('Sending data:', this.email, this.password);

    this.http.post('http://127.0.0.1:8000/login', {
      email: this.email,
      password: this.password
    }).subscribe({
      next: (res: any) => {
        alert(res.message || 'Login Successful');
        this.router.navigate(['/audio-record']);
      },
      error: (err) => {
        console.error(err);
        alert('Invalid login');
      }
    });

}

}
