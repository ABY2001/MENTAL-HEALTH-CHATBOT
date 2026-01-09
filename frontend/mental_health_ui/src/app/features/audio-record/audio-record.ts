import {
  Component,
  OnDestroy,
  ViewChild,
  ElementRef,
  AfterViewChecked,
  ChangeDetectorRef
} from '@angular/core';
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
  recordingStartTime = 0;
  recordingTime = '0:00';
  recordingInterval: any = null;

  // Audio preview state
  showAudioPreview = false;
  audioURL: string | null = null;
  currentAudioBlob: Blob | null = null;

  // Emotion state
  currentEmotion: string | null = null;
  emotionStatus = '😊 Ready to chat';

  private shouldScrollToBottom = false;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef
  ) {}

  ngAfterViewChecked() {
    if (this.shouldScrollToBottom) {
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }
  }

  ngOnDestroy() {
    if (this.recordingInterval) clearInterval(this.recordingInterval);
    if (this.isRecording) this.stopRecording();
  }

  getCurrentTime(): string {
    const now = new Date();
    return now.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
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
    if (this.messagesContainer) {
      this.messagesContainer.nativeElement.scrollTop =
        this.messagesContainer.nativeElement.scrollHeight;
    }
  }

  onKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  // ===================== TEXT MESSAGE =====================
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
          if (lastMessage.isUser) lastMessage.emotion = emotion;

          this.currentEmotion = emotion;
          this.updateEmotionStatus(emotion);

          setTimeout(() => {
            this.isTyping = false;
            this.addMessage(botResponse, false);

            // ⚠️ Safety / crisis warning
            if (res.safety?.warning) {
              setTimeout(() => {
                this.addMessage(`⚠️ ${res.safety.warning}`, false);
              }, 500);
            }
          }, 1000);
        },
        error: () => {
          this.isTyping = false;
          this.addMessage(
            "I'm having trouble understanding right now. Please try again.",
            false
          );
        }
      });
  }

  updateEmotionStatus(emotion: string) {
    const emojis: any = {
      happy: '😊',
      calm: '😌',
      sad: '😔',
      angry: '😠',
      fearful: '😰',
      neutral: '😐',
      disgust: '😖',
      surprised: '😲'
    };
    this.emotionStatus = `${emojis[emotion] || '💙'} Feeling ${emotion}`;
  }

  // ===================== AUDIO RECORDING =====================
  async toggleRecording() {
    this.isRecording ? this.stopRecording() : await this.startRecording();
  }

  async startRecording() {
    try {
      this.audioChunks = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);

      this.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

        this.currentAudioBlob = audioBlob;
        this.audioURL = URL.createObjectURL(audioBlob);
        this.showAudioPreview = true;

        stream.getTracks().forEach(t => t.stop());

        // 🔥 Fix: force UI update immediately
        this.cdr.detectChanges();
      };

      this.mediaRecorder.start();
      this.isRecording = true;

      this.recordingStartTime = Date.now();
      this.recordingInterval = setInterval(
        () => this.updateRecordingTime(),
        100
      );
    } catch {
      alert('Please allow microphone access to use voice messages.');
    }
  }

  stopRecording() {
    if (this.mediaRecorder?.state === 'recording') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      if (this.recordingInterval) clearInterval(this.recordingInterval);
    }
  }

  updateRecordingTime() {
    const elapsed = Date.now() - this.recordingStartTime;
    const s = Math.floor(elapsed / 1000);
    this.recordingTime = `${Math.floor(s / 60)}:${(s % 60)
      .toString()
      .padStart(2, '0')}`;
  }

  // ===================== SEND AUDIO =====================
  sendAudio(audioBlob: Blob) {
    this.addMessage('🎤 Voice message', true);
    this.isTyping = true;

    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');

    this.http.post<any>('http://127.0.0.1:8000/predict-emotion', formData)
      .subscribe({
        next: (res) => {
          const emotion = res.emotion;

          const lastMessage = this.messages[this.messages.length - 1];
          if (lastMessage.isUser) lastMessage.emotion = emotion;

          this.currentEmotion = emotion;
          this.updateEmotionStatus(emotion);

          setTimeout(() => {
            this.isTyping = false;
            this.addMessage(res.bot_response, false);
          }, 1000);
        },
        error: () => {
          this.isTyping = false;
          this.addMessage(
            'Error analyzing your voice. Please try again.',
            false
          );
        }
      });
  }

  cancelRecording() {
    if (this.audioURL) URL.revokeObjectURL(this.audioURL);
    this.audioURL = null;
    this.currentAudioBlob = null;
    this.showAudioPreview = false;
    this.recordingTime = '0:00';
  }

  confirmSendAudio() {
    if (!this.currentAudioBlob) return;

    this.showAudioPreview = false;
    this.sendAudio(this.currentAudioBlob);

    if (this.audioURL) URL.revokeObjectURL(this.audioURL);
    this.audioURL = null;
    this.currentAudioBlob = null;
  }

  getEmotionClass(emotion: string): string {
    return `emotion-${emotion.toLowerCase()}`;
  }
}
