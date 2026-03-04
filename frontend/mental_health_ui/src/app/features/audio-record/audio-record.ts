// audio-record.component.ts - UPDATED: Captures video emotion properly

import { Component, OnDestroy, ViewChild, ElementRef, AfterViewChecked, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { EmotionWidget } from '../emotion-widget/emotion-widget';

interface Message {
  text: string;
  isUser: boolean;
  emotion?: string;
  time: string;
}

interface ChatSession {
  id?: number;
  user_message: string;
  bot_response: string;
  emotion: string;
  emotion_confidence: number;
  created_at: string;
}

@Component({
  selector: 'app-audio-record',
  standalone: true,
  imports: [CommonModule, FormsModule, EmotionWidget],
  templateUrl: './audio-record.html',
  styleUrl: './audio-record.css'
})
export class AudioRecord implements OnDestroy, AfterViewChecked {
  @ViewChild('messagesContainer') messagesContainer!: ElementRef;
  @ViewChild('videoPreview') videoPreview!: ElementRef;
  
  currentUserId: number | null = null;
  
  messages: Message[] = [
    {
      text: "Hello! I'm here to support you. How are you feeling today? You can type, speak, or show your face to me.",
      isUser: false,
      time: this.getCurrentTime()
    }
  ];
  messageInput = '';
  isTyping = false;
  
  // Chat history
  showChatHistory = false;
  chatHistory: ChatSession[] = [];
  chatHistoryLoading = false;
  
  // Audio recording
  isRecording = false;
  mediaRecorder!: MediaRecorder;
  audioChunks: Blob[] = [];
  recordingStartTime: number = 0;
  recordingTime = '0:00';
  recordingInterval: any = null;
  audioStream: MediaStream | null = null;
  
  // Audio preview
  showAudioPreview = false;
  audioURL: string | null = null;
  currentAudioBlob: Blob | null = null;
  audioPreviewDuration = '0:00';
  
  // ✅ VIDEO EMOTION TRACKING
  isVideoActive = false;
  videoStream: MediaStream | null = null;
  videoElement: HTMLVideoElement | null = null;
  faceAnalysisInterval: any = null;
  currentFaceEmotion: { emotion: string; confidence: number } | null = null;
  
  emotionStatus = '😊 Ready to chat';
  currentEmotion: string | null = null;
  showDeleteToast:boolean = false;
  
  private shouldScrollToBottom = false;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef 
  ) {
    const userIdStr = localStorage.getItem('user_id');
    this.currentUserId = userIdStr ? parseInt(userIdStr, 10) : null;
  }

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
    this.stopVideoAnalysis();
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
    this.cdr.detectChanges();
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

  // ==================== VIDEO EMOTION TRACKING ====================
  async startVideoAnalysis() {
    if (this.isVideoActive) return;

    try {
      console.log('🎥 Starting video emotion tracking...');
      
      this.videoStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      this.isVideoActive = true;
      this.cdr.markForCheck();
      
      this.videoElement = document.createElement('video');
      this.videoElement.srcObject = this.videoStream;
      this.videoElement.play();
      
      this.startContinuousEmotionAnalysis();
      
      console.log('✓ Video emotion tracking started (SILENT)');
      
    } catch (error) {
      console.error(' Camera error:', error);
      alert('Please allow camera access');
      this.isVideoActive = false;
    }
  }

  stopVideoAnalysis() {
  if (!this.isVideoActive) return;

  console.log('\n Stopping video emotion tracking...');
  
  if (this.faceAnalysisInterval) {
    clearInterval(this.faceAnalysisInterval);
    this.faceAnalysisInterval = null;
  }
  
  if (this.videoStream) {
    this.videoStream.getTracks().forEach(track => {
      track.stop();
    });
    this.videoStream = null;
  }
  
  this.videoElement = null;
  this.isVideoActive = false;
  
  //  Store last emotion before clearing (for reference)
  const lastEmotion = this.currentFaceEmotion;
  this.currentFaceEmotion = null;
  
  this.cdr.markForCheck();
  
  console.log(`✓ Video tracking stopped (last emotion: ${JSON.stringify(lastEmotion)})`);
}

  toggleVideoAnalysis() {
    if (this.isVideoActive) {
      this.stopVideoAnalysis();
    } else {
      this.startVideoAnalysis();
    }
  }

  // ==================== CONTINUOUS EMOTION ANALYSIS ====================
  startContinuousEmotionAnalysis() {
    this.faceAnalysisInterval = setInterval(async () => {
      if (!this.videoElement || !this.isVideoActive) return;

      try {
        const canvas = document.createElement('canvas');
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        
        if (ctx && this.videoElement.videoWidth > 0) {
          ctx.drawImage(this.videoElement, 0, 0);
          const frameBase64 = canvas.toDataURL('image/jpeg');
          this.analyzeFrameEmotion(frameBase64);
        }
      } catch (error) {
        console.error('Frame capture error:', error);
      }
    }, 500);
  }

  // ==================== ANALYZE FRAME EMOTION ====================
  analyzeFrameEmotion(frameBase64: string) {
    this.http.post<any>('http://127.0.0.1:8000/analyze-frame-emotion', {
      frame: frameBase64
    }).subscribe({
      next: (res) => {
        this.currentFaceEmotion = {
          emotion: res.emotion,
          confidence: res.confidence
        };
        console.log(`📷 Face emotion: ${res.emotion} (${res.confidence.toFixed(2)})`);
      },
      error: (err) => {
        console.debug('Frame analysis skipped');
      }
    });
  }

  // ==================== CHAT HISTORY ====================
  loadChatHistory() {
    if (!this.currentUserId) {
      alert('Please log in first');
      return;
    }

    this.chatHistoryLoading = true;
    this.showChatHistory = true;

    this.http.get<any>(`http://127.0.0.1:8000/chat-history/${this.currentUserId}`)
      .subscribe({
        next: (res) => {
          this.chatHistory = res.messages || [];
          this.chatHistoryLoading = false;
          this.cdr.detectChanges();
        },
        error: (err) => {
          this.chatHistoryLoading = false;
          alert('Failed to load chat history');
        }
      });
  }

  loadPreviousChat(chat: ChatSession) {
    this.showChatHistory = false;
    this.messages = [];
    const timestamp = new Date(chat.created_at).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });

    this.messages.push({
      text: chat.user_message,
      isUser: true,
      emotion: chat.emotion,
      time: timestamp
    });

    this.messages.push({
      text: chat.bot_response,
      isUser: false,
      time: timestamp
    });

    this.shouldScrollToBottom = true;
    this.cdr.detectChanges();
  }

  deleteAllChats() {
    if (!this.currentUserId) return;
    const confirm = window.confirm('Delete all chat history?');
    if (!confirm) return;

    this.http.delete<any>(`http://127.0.0.1:8000/delete-chat-session/${this.currentUserId}`)
      .subscribe({
        next: (res) => {
          this.chatHistory = [];
          alert('Chat history deleted');
        },
        error: (err) => alert('Failed to delete')
      });
  }

  closeChatHistory() {
    this.showChatHistory = false;
  }

  deleteIndividualChat(chatId: number | undefined, chatPreview: string) {
    if (!chatId) {
      alert('Error: Chat ID not found');
      return;
    }

    const confirm = window.confirm(`Delete this chat?`);
    if (!confirm) return;

    this.http.delete<any>(`http://127.0.0.1:8000/delete-chat-message/${chatId}`)
      .subscribe({
        next: (res) => {
          this.chatHistory = this.chatHistory.filter(chat => chat.id !== chatId);
          alert('Chat deleted');
          this.newChat();
        },
        error: (err) => alert('Failed to delete'),
      });
  }

  // ==================== TEXT MESSAGE WITH VIDEO EMOTION ====================
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

    const payload = {
      text: text,
      video_emotion: this.currentFaceEmotion
    };

    console.log(`📤 Sending text with face emotion: ${this.currentFaceEmotion?.emotion || 'none'}`);

    this.http.post<any>('http://127.0.0.1:8000/predict-emotion-text-with-video', payload)
      .subscribe({
        next: (res) => {
          const emotion = res.emotion;
          const botResponse = res.bot_response;
          
          console.log(`✅ Response - Text: ${res.text_emotion}, Video: ${res.video_emotion}, Fused: ${emotion}`);
          
          const lastMessage = this.messages[this.messages.length - 1];
          if (lastMessage.isUser) {
            lastMessage.emotion = emotion;
          }
          
          this.currentEmotion = emotion;
          this.updateEmotionStatus(emotion);
          
          setTimeout(() => {
            this.isTyping = false;
            this.addMessage(botResponse, false);
            this.saveChatToHistory(text, botResponse, emotion, res.confidence);
          }, 1000);
        },
        error: (err) => {
          console.error('Error:', err);
          this.isTyping = false;
          this.addMessage("Error. Please try again.", false);
        }
      });
  }

  updateEmotionStatus(emotion: string) {
    const emotionEmojis: { [key: string]: string } = {
      happy: '😊', calm: '😌', sad: '😔', angry: '😠', 
      fearful: '😰', neutral: '😐', disgust: '😖', surprised: '😲'
    };
    this.emotionStatus = `${emotionEmojis[emotion] || '💙'} Feeling ${emotion}`;
  }

  // ==================== AUDIO RECORDING ====================
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
      
      this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(this.audioStream);

      this.mediaRecorder.ondataavailable = event => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        this.currentAudioBlob = audioBlob;
        this.audioURL = URL.createObjectURL(audioBlob);
        this.audioPreviewDuration = this.recordingTime;
        this.showAudioPreview = true;
        this.cdr.markForCheck();
        
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
      this.cdr.markForCheck();
      
      this.recordingInterval = setInterval(() => {
        this.updateRecordingTime();
        this.cdr.markForCheck();
      }, 100);

    } catch (error) {
      console.error('Microphone error:', error);
      alert('Please allow microphone access.');
      this.isRecording = false;
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      
      if (this.recordingInterval) {
        clearInterval(this.recordingInterval);
        this.recordingInterval = null;
      }
      
      this.cdr.markForCheck();
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
  }

  // ==================== SEND AUDIO WITH VIDEO EMOTION ====================
sendAudio(audioBlob: Blob, capturedVideoEmotion?: any) {
  console.log('\n\n╔════════════════════════════════════════════╗');
  console.log('║    SENDING AUDIO - VIDEO STAYS ACTIVE      ║');
  console.log('╚════════════════════════════════════════════╝');
  
  // ✅ Use captured emotion (don't rely on currentFaceEmotion which might change)
  const videoEmotionToSend = capturedVideoEmotion;
  
  console.log(`\n📊 PARAMETERS:`);
  console.log(`   audioBlob: ${audioBlob ? audioBlob.size + ' bytes' : 'null'}`);
  console.log(`   capturedVideoEmotion: ${JSON.stringify(capturedVideoEmotion)}`);
  console.log(`   videoEmotionToSend: ${JSON.stringify(videoEmotionToSend)}`);
  console.log(`   isVideoActive: ${this.isVideoActive}`);
  
  this.addMessage('🎤 Voice message', true);
  
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.webm');
  
  // ✅ Add captured video emotion
  if (videoEmotionToSend && videoEmotionToSend.emotion) {
    console.log(`\n✅ ADDING VIDEO EMOTION:`);
    console.log(`   emotion: ${videoEmotionToSend.emotion}`);
    console.log(`   confidence: ${videoEmotionToSend.confidence}`);
    formData.append('video_emotion', JSON.stringify(videoEmotionToSend));
  } else {
    console.log(`\n⚠️ NO VIDEO EMOTION TO SEND`);
  }

  console.log(`\n📤 POST to /predict-emotion-with-video`);
  
  this.isTyping = true;

  this.http.post<any>('http://127.0.0.1:8000/predict-emotion-with-video', formData)
    .subscribe({
      next: (res) => {
        console.log('\n✅ RESPONSE RECEIVED:');
        console.log(`   🎤 Audio: ${res.audio_emotion}`);
        console.log(`   📝 Text: ${res.text_emotion}`);
        console.log(`   📷 Video: ${res.video_emotion}`);
        console.log(`   ✅ Fused: ${res.emotion}`);
        
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
          
          const displayText = transcription || '🎤 Voice message';
          this.saveChatToHistory(displayText, botResponse, emotion, res.confidence);
          
          // ✅ NOW stop video (after response received)
          this.stopVideoAnalysis();
        }, 1000);
      },
      error: (err) => {
        console.error('❌ ERROR:', err);
        this.isTyping = false;
        this.addMessage('Error analyzing. Please try again.', false);
        this.stopVideoAnalysis();
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
    // ✅ CAPTURE emotion BEFORE ANYTHING
    const capturedEmotion = this.currentFaceEmotion ? 
      JSON.parse(JSON.stringify(this.currentFaceEmotion)) : null;
    
    console.log(`\n🎯 SEND AUDIO:`);
    console.log(`   Captured emotion: ${JSON.stringify(capturedEmotion)}`);
    console.log(`   Is video active: ${this.isVideoActive}`);
    
    const blobToSend = this.currentAudioBlob;
    
    // ✅ CRITICAL: DO NOT STOP VIDEO YET!
    // Video will be stopped AFTER sending completes
    
    this.clearPreviousRecording();
    this.sendAudio(blobToSend, capturedEmotion);
    
    // ✅ Video stays running until after response
  }
}
  // ==================== SAVE CHAT ====================
  private saveChatToHistory(userMessage: string, botResponse: string, emotion: string, confidence: number) {
    if (!this.currentUserId) return;

    const payload = {
      user_id: this.currentUserId,
      user_message: userMessage,
      bot_response: botResponse,
      emotion: emotion,
      emotion_confidence: confidence
    };
   
    this.http.post<any>('http://127.0.0.1:8000/save-chat', payload)
      .subscribe({
        next: (res) => {
          console.log('✓ Chat saved');
        },
        error: (err) => {
          console.error('Error saving chat:', err);
        }
      });
  }

  // ==================== NEW CHAT ====================
  newChat() {
    // const confirm = window.confirm('Start a new conversation?');
    // if (!confirm) return;

    this.messages = [
      {
        text: "Hello! I'm here to support you. How are you feeling today?",
        isUser: false,
        time: this.getCurrentTime()
      }
    ];

    this.messageInput = '';
    this.currentEmotion = null;
    this.emotionStatus = '😊 Ready to chat';
    this.showAudioPreview = false;
    this.showChatHistory = false;
    this.stopVideoAnalysis();
    this.isRecording = false;

    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
      this.recordingInterval = null;
    }

    console.log('✓ New chat started');
    this.cdr.detectChanges();
  }
}