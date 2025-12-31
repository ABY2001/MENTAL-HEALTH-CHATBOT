// audio-record.component.ts
import { Component, OnDestroy, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Sidebar } from '../../shared/components/sidebar/sidebar';

interface Message {
  text: string;
  isUser: boolean;
  emotion?: string;
  time: string;
}

@Component({
  selector: 'app-audio-record',
  standalone: true,
  imports: [CommonModule, FormsModule, Sidebar],
  templateUrl: './audio-record.html',
  styleUrl: './audio-record.css'
})
export class AudioRecord implements OnDestroy, AfterViewChecked {
  @ViewChild('messagesContainer') messagesContainer!: ElementRef;
  
  // Chat state
  messages: Message[] = [
    {
      text: "Hello! I'm here to support you. How are you feeling today? You can type or speak to me.",
      isUser: false,
      time: this.getCurrentTime()
    }
  ];
  messageInput = '';
  isTyping = false;
  
  // Recording state
  isRecording = false;
  mediaRecorder!: MediaRecorder;
  audioChunks: Blob[] = [];
  recordingStartTime: number = 0;
  recordingTime = '0:00';
  recordingInterval: any = null;
  
  // Emotion state
  currentEmotion: string | null = null;
  emotionStatus = '😊 Ready to chat';
  
  private shouldScrollToBottom = false;

  constructor(private http: HttpClient) {}

  ngAfterViewChecked() {
    if (this.shouldScrollToBottom) {
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }
  }

  ngOnDestroy() {
    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
    }
    if (this.isRecording) {
      this.stopRecording();
    }
  }

  getCurrentTime(): string {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  }

  addMessage(text: string, isUser: boolean, emotion?: string) {
    this.messages.push({
      text,
      isUser,
      emotion,
      time: this.getCurrentTime()
    });
    this.shouldScrollToBottom = true;
  }

  scrollToBottom() {
    try {
      if (this.messagesContainer) {
        this.messagesContainer.nativeElement.scrollTop = 
          this.messagesContainer.nativeElement.scrollHeight;
      }
    } catch (err) {
      console.error('Scroll error:', err);
    }
  }

  onKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  sendMessage() {
    const text = this.messageInput.trim();
    if (!text) return;

    // Add user message
    this.addMessage(text, true);
    this.messageInput = '';

    // Show typing indicator
    this.isTyping = true;

    // Send to backend for emotion detection
    this.http.post<any>('http://127.0.0.1:8000/predict-emotion-text', { text })
      .subscribe({
        next: (res) => {
          const emotion = res.emotion;
          const confidence = Math.round(res.confidence * 100);
          
          // Update last user message with emotion
          const lastMessage = this.messages[this.messages.length - 1];
          if (lastMessage.isUser) {
            lastMessage.emotion = emotion;
          }
          
          this.currentEmotion = emotion;
          this.updateEmotionStatus(emotion);
          
          // Get bot response
          setTimeout(() => {
            const response = this.getBotResponse(emotion, text);
            this.isTyping = false;
            this.addMessage(response, false);
          }, 1000);
        },
        error: (err) => {
          console.error('Error detecting emotion:', err);
          this.isTyping = false;
          this.addMessage("I'm having trouble understanding right now. Please try again.", false);
        }
      });
  }

  updateEmotionStatus(emotion: string) {
    const emotionEmojis: { [key: string]: string } = {
      happy: '😊',
      calm: '😌',
      sad: '😔',
      angry: '😠',
      fearful: '😰',
      neutral: '😐',
      disgust: '😖',
      surprised: '😲'
    };
    
    this.emotionStatus = `${emotionEmojis[emotion] || '💙'} Feeling ${emotion}`;
  }

  getBotResponse(emotion: string, userMessage: string): string {
    const responses: { [key: string]: string } = {
      happy: "That's wonderful! I'm glad you're feeling positive. What's making you happy today?",
      calm: "It's great that you're feeling calm. Would you like to talk about what's on your mind?",
      sad: "I hear that you're feeling down. Remember, it's okay to feel this way. Would you like to share what's troubling you?",
      angry: "I understand you're feeling frustrated. It's completely valid. Let's work through this together. What's bothering you?",
      fearful: "I sense you might be worried or anxious. Remember, you're not alone. Would you like to talk about what's concerning you?",
      neutral: "I'm here to listen. Feel free to share whatever is on your mind.",
      disgust: "I understand you're feeling uncomfortable. Would you like to talk about what's bothering you?",
      surprised: "That sounds unexpected! Would you like to share more about what happened?"
    };
    
    return responses[emotion] || "Thank you for sharing. How can I support you today?";
  }

  async toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      await this.startRecording();
    }
  }

  async startRecording() {
    try {
      this.audioChunks = [];
      
      console.log('Requesting microphone access...');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone access granted');
      
      this.mediaRecorder = new MediaRecorder(stream);
      console.log('MediaRecorder created');

      this.mediaRecorder.ondataavailable = event => {
        console.log('Audio chunk received:', event.data.size, 'bytes');
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = () => {
        console.log('Recording stopped, total chunks:', this.audioChunks.length);
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        console.log('Audio blob created:', audioBlob.size, 'bytes');
        this.sendAudio(audioBlob);
        stream.getTracks().forEach(track => {
          track.stop();
          console.log('Track stopped');
        });
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      console.log('Recording started');
      
      this.recordingStartTime = Date.now();
      this.recordingInterval = setInterval(() => {
        this.updateRecordingTime();
      }, 100);

    } catch (error) {
      console.error('Microphone access error:', error);
      alert('Please allow microphone access to use voice messages.');
    }
  }

  stopRecording() {
    console.log('Stopping recording...');
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      
      if (this.recordingInterval) {
        clearInterval(this.recordingInterval);
      }
      console.log('Recording stopped successfully');
    }
  }

  updateRecordingTime() {
    const elapsed = Date.now() - this.recordingStartTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    this.recordingTime = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }

  sendAudio(audioBlob: Blob) {
    // Add voice message indicator
    this.addMessage('🎤 Voice message', true);
    
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    this.isTyping = true;

    this.http.post<any>('http://127.0.0.1:8000/predict-emotion', formData)
      .subscribe({
        next: (res) => {
          const emotion = res.emotion;
          const confidence = Math.round(res.confidence * 100);
          
          // Update last user message with emotion
          const lastMessage = this.messages[this.messages.length - 1];
          if (lastMessage.isUser) {
            lastMessage.emotion = emotion;
          }
          
          this.currentEmotion = emotion;
          this.updateEmotionStatus(emotion);
          
          // Get bot response
          setTimeout(() => {
            const response = this.getBotResponse(emotion, 'voice message');
            this.isTyping = false;
            this.addMessage(response, false);
          }, 1000);
        },
        error: (err) => {
          console.error('Error analyzing emotion:', err);
          this.isTyping = false;
          this.addMessage('Error analyzing your voice. Please try again.', false);
        }
      });
  }

  getEmotionClass(emotion: string): string {
    return `emotion-${emotion.toLowerCase()}`;
  }
}