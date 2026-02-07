// audio-record.component.ts - FIXED TIMER DISPLAY

import { Component, OnDestroy, ViewChild, ElementRef, AfterViewChecked, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

interface Message {
  text: string;
  isUser: boolean;
  emotion?: string;
  time: string;
}

@Component({
  selector: 'app-audio-record',
  standalone: true,
  imports: [CommonModule, FormsModule],
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
  audioStream: MediaStream | null = null;
  
  // Audio preview state
  showAudioPreview = false;
  audioURL: string | null = null;
  currentAudioBlob: Blob | null = null;
  audioPreviewDuration = '0:00'; // ⭐ Separate duration for modal
  
  // Emotion state
  currentEmotion: string | null = null;
  emotionStatus = '😊 Ready to chat';
  
  private shouldScrollToBottom = false;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef // ⭐ Inject ChangeDetectorRef
  ) {}

  ngAfterViewChecked() {
    if (this.shouldScrollToBottom) {
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }
  }

  ngOnDestroy() {
    this.cleanup();
  }

  private cleanup() {
    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
    }
    if (this.isRecording) {
      this.stopRecording();
    }
    if (this.audioURL) {
      URL.revokeObjectURL(this.audioURL);
    }
    if (this.audioStream) {
      this.audioStream.getTracks().forEach(track => {
        track.stop();
      });
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

    this.addMessage(text, true);
    this.messageInput = '';
    this.isTyping = true;

    this.http.post<any>('http://127.0.0.1:8000/predict-emotion-text', { text })
      .subscribe({
        next: (res) => {
          const emotion = res.emotion;
          const botResponse = res.bot_response;
          
          const lastMessage = this.messages[this.messages.length - 1];
          if (lastMessage.isUser) {
            lastMessage.emotion = emotion;
          }
          
          this.currentEmotion = emotion;
          this.updateEmotionStatus(emotion);
          
          setTimeout(() => {
            this.isTyping = false;
            this.addMessage(botResponse, false);
            
            if (res.safety?.warning) {
              setTimeout(() => {
                this.addMessage(`⚠️ ${res.safety.warning}`, false);
              }, 500);
            }
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

  async toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      await this.startRecording();
    }
  }

  async startRecording() {
    try {
      this.clearPreviousRecording();
      
      this.audioChunks = [];
      console.log('Requesting microphone access...');
      
      this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone access granted');
      
      this.mediaRecorder = new MediaRecorder(this.audioStream);

      this.mediaRecorder.ondataavailable = event => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        this.currentAudioBlob = audioBlob;
        this.audioURL = URL.createObjectURL(audioBlob);
        
        // ⭐ Set modal duration to final recording time
        this.audioPreviewDuration = this.recordingTime;
        
        this.showAudioPreview = true;
        this.cdr.markForCheck(); // ⭐ Force change detection
        
        if (this.audioStream) {
          this.audioStream.getTracks().forEach(track => {
            track.stop();
          });
        }
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.recordingStartTime = Date.now();
      this.recordingTime = '0:00';
      this.cdr.markForCheck(); // ⭐ Force change detection
      
      // ⭐ Update timer every 100ms with change detection
      this.recordingInterval = setInterval(() => {
        this.updateRecordingTime();
        this.cdr.markForCheck(); // ⭐ Trigger UI update
      }, 100);

    } catch (error) {
      console.error('Microphone access error:', error);
      alert('Please allow microphone access to use voice messages.');
      this.isRecording = false;
    }
  }

  stopRecording() {
    console.log('Stopping recording...');
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      
      if (this.recordingInterval) {
        clearInterval(this.recordingInterval);
        this.recordingInterval = null;
      }
      
      this.cdr.markForCheck(); // ⭐ Force change detection
    }
  }

  updateRecordingTime() {
    const elapsed = Date.now() - this.recordingStartTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    this.recordingTime = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }

  private clearPreviousRecording() {
    if (this.audioURL) {
      URL.revokeObjectURL(this.audioURL);
      this.audioURL = null;
    }
    this.currentAudioBlob = null;
    this.showAudioPreview = false;
    this.recordingTime = '0:00';
    this.audioPreviewDuration = '0:00';
    console.log('Previous recording cleared');
  }

  sendAudio(audioBlob: Blob) {
    this.addMessage('🎤 Voice message', true);
    
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');

    this.isTyping = true;

    this.http.post<any>('http://127.0.0.1:8000/predict-emotion', formData)
      .subscribe({
        next: (res) => {
          console.log('Response received:', res);
          const emotion = res.emotion;
          const transcription = res.transcription || '';
          const botResponse = res.bot_response;
          
          const lastMessage = this.messages[this.messages.length - 1];
          if (lastMessage.isUser) {
            if (transcription) {
              lastMessage.text = `🎤 "${transcription}"`;
            }
            lastMessage.emotion = emotion;
          }
          
          this.currentEmotion = emotion;
          this.updateEmotionStatus(emotion);
          
          setTimeout(() => {
            this.isTyping = false;
            this.addMessage(botResponse, false);
            
            if (res.audio_quality && !res.audio_quality.clear) {
              setTimeout(() => {
                this.addMessage(`⚠️ ${res.audio_quality.issue}`, false);
              }, 500);
            }
            
            if (res.safety?.warning) {
              setTimeout(() => {
                this.addMessage(`⚠️ ${res.safety.warning}`, false);
              }, 1000);
            }
          }, 1000);
        },
        error: (err) => {
          console.error('Error analyzing emotion:', err);
          this.isTyping = false;
          
          let errorMsg = 'Error analyzing your voice. Please try again.';
          if (err.error?.detail) {
            errorMsg = `Error: ${err.error.detail}`;
          }
          
          this.addMessage(errorMsg, false);
        }
      });
  }

  getEmotionClass(emotion: string): string {
    return `emotion-${emotion.toLowerCase()}`;
  }

  cancelRecording() {
    this.clearPreviousRecording();
  }

  confirmSendAudio() {
    if (this.currentAudioBlob) {
      const blobToSend = this.currentAudioBlob;
      this.clearPreviousRecording();
      this.sendAudio(blobToSend);
    }
  }
}