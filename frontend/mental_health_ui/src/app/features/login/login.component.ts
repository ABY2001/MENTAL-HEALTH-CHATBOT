import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
  imports: [CommonModule, FormsModule],
  standalone: true
})
export class LoginComponent implements OnInit {
  password: string = '';
  email: string = '';
  isLoading: boolean = false;
  errorMessage: string = '';

  constructor(private http: HttpClient, private router: Router) { }

  ngOnInit() {
  }

  getLoginCredentials() {
    if (!this.email || !this.password) {
      this.errorMessage = 'Please enter email and password';
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';

    console.log('Sending data:', this.email, this.password);

    this.http.post('http://127.0.0.1:8000/login', {
      email: this.email,
      password: this.password
    }).subscribe({
      next: (res: any) => {
        console.log('✓ Login successful:', res);
        
        localStorage.setItem('user_id', res.user_id.toString());
        localStorage.setItem('email', res.email);
        
        console.log('✓ Saved to localStorage - user_id:', res.user_id);
        
        this.isLoading = false;
        this.router.navigate(['/audio-record']);
      },
      error: (err) => {
        console.error('❌ Login error:', err);
        this.isLoading = false;
        this.errorMessage = 'Invalid login credentials';
        alert('Invalid login');
      }
    });
  }
}