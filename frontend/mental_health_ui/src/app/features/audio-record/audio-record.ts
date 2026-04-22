// audio-record.component.ts - UPDATED: Session-based chat history with proper session management

import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewChecked, ChangeDetectorRef } from '@angular/core';
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

interface ChatMessage {
  id: number;
  turn_number: number;
  user_message: string;
  bot_response: string;
  emotion: string;
  emotion_confidence: number;
  created_at: string;
}

interface ChatSession {
  session_id: string;
  message_count: number;
  first_message: string;
  last_emotion: string;
  created_at: string;
  updated_at: string;
  all_messages: ChatMessage[];
}

@Component({
  selector: 'app-audio-record',
  standalone: true,
  imports: [CommonModule, FormsModule, EmotionWidget],
  templateUrl: './audio-record.html',
  styleUrl: './audio-record.css'
})
export class AudioRecord implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('messagesContainer') messagesContainer!: ElementRef;
  
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
  
  // ⭐ Chat history - Sessions instead of individual messages
  showChatHistory = false;
  chatSessions: ChatSession[] = [];  // ⭐ CHANGED from chatHistory to chatSessions
  chatHistoryLoading = false;
  
  // ⭐ Session management
  currentSessionId: string | null = null;
  
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
  
  // Video emotion tracking
  isVideoActive = false;
  videoStream: MediaStream | null = null;
  videoElement: HTMLVideoElement | null = null;
  faceAnalysisInterval: any = null;
  currentFaceEmotion: { emotion: string; confidence: number } | null = null;
  
  emotionStatus = '😊 Ready to chat';
  currentEmotion: string | null = null;
  showDeleteToast: boolean = false;
  
  private shouldScrollToBottom = false;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef 
  ) {
    const userIdStr = localStorage.getItem('user_id');
    this.currentUserId = userIdStr ? parseInt(userIdStr, 10) : null;
  }

  ngOnInit() {
    // ⭐ Generate session ID
    this.currentSessionId = this.generateSessionId();
    console.log('✓ Session ID initialized:', this.currentSessionId);
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

  // ⭐ Generate unique session ID
  private generateSessionId(): string {
    const stored = localStorage.getItem('currentSessionId');
    if (stored) {
      return stored;
    }
    const newSession = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('currentSessionId', newSession);
    return newSession;
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
      console.error('Camera error:', error);
      alert('Please allow camera access');
      this.isVideoActive = false;
    }
  }

  stopVideoAnalysis() {
    if (!this.isVideoActive) return;

    console.log('Stopping video emotion tracking...');
    
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
    this.currentFaceEmotion = null;
    
    this.cdr.markForCheck();
    
    console.log('✓ Video tracking stopped');
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
        console.log(`📷 Face emotion: ${res.emotion} (${(res.confidence * 100).toFixed(0)}%)`);
      },
      error: (err) => {
        console.debug('Frame analysis skipped');
      }
    });
  }

  // ==================== CHAT HISTORY - SESSIONS ====================
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
          console.log('✓ Chat history received:', res);
          // ⭐ Changed: Load sessions instead of individual messages
          this.chatSessions = res.sessions || [];
          console.log(`✓ Loaded ${this.chatSessions.length} sessions`);
          this.chatHistoryLoading = false;
          this.cdr.detectChanges();
        },
        error: (err) => {
          console.error('❌ Error loading chat history:', err);
          this.chatHistoryLoading = false;
          alert('Failed to load chat history');
        }
      });
  }

  // ⭐ Load entire session (not individual message)
  loadPreviousChat(session: ChatSession) {
    console.log('✓ Loading session:', session.session_id);
    this.showChatHistory = false;
    this.messages = [];
    
    // Reconstruct messages from session.all_messages
    if (session.all_messages && Array.isArray(session.all_messages)) {
      for (let msg of session.all_messages) {
        const timestamp = new Date(msg.created_at).toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit'
        });

        // Add user message
        this.messages.push({
          text: msg.user_message,
          isUser: true,
          emotion: msg.emotion,
          time: timestamp
        });

        // Add bot response
        this.messages.push({
          text: msg.bot_response,
          isUser: false,
          time: timestamp
        });
      }
    }

    // ⭐ Set current session to this one
    this.currentSessionId = session.session_id;
    localStorage.setItem('currentSessionId', this.currentSessionId);

    this.shouldScrollToBottom = true;
    this.cdr.detectChanges();
    
    console.log(`✓ Loaded ${this.messages.length} messages from session`);
  }

  deleteAllChats() {
    if (!this.currentUserId) return;
    
    if (!window.confirm('Delete ALL chat history?')) return;

    this.http.delete<any>(`http://127.0.0.1:8000/delete-all-chats/${this.currentUserId}`)
      .subscribe({
        next: (res) => {
          console.log('✓ All chats deleted');
          this.chatSessions = [];
          this.showDeleteToast = true;
          setTimeout(() => this.showDeleteToast = false, 3000);
          this.newChat();
        },
        error: (err) => {
          console.error('Error deleting chats:', err);
          alert('Failed to delete chats');
        }
      });
  }

  closeChatHistory() {
    this.showChatHistory = false;
  }

  // ⭐ Delete session (not individual message)
  deleteIndividualChat(sessionId: string, firstMessage: string) {
    if (!window.confirm('Delete this entire conversation?')) return;

    this.http.delete<any>(`http://127.0.0.1:8000/delete-chat-session/${sessionId}`)
      .subscribe({
        next: (res) => {
          console.log('✓ Session deleted');
          this.chatSessions = this.chatSessions.filter(s => s.session_id !== sessionId);
          this.showDeleteToast = true;
          this.cdr.detectChanges();
          setTimeout(() => this.showDeleteToast = false, 3000);
        },
        error: (err) => {
          console.error('Error deleting session:', err);
          alert('Failed to delete chat');
        }
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
            // ⭐ Save with session_id
            this.saveChatToHistory(text, botResponse, emotion, res.emotion_confidence || 0);
          }, 1000);
        },
        error: (err) => {
          console.error('Error:', err);
          this.isTyping = false;
          this.addMessage('Error. Please try again.', false);
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
  confirmSendAudio() {
    if (this.currentAudioBlob) {
      // ⭐ Capture emotion before anything
      const capturedEmotion = this.currentFaceEmotion ? 
        JSON.parse(JSON.stringify(this.currentFaceEmotion)) : null;
      
      const blobToSend = this.currentAudioBlob;
      
      this.clearPreviousRecording();
      this.sendAudio(blobToSend, capturedEmotion);
    }
  }

  sendAudio(audioBlob: Blob, capturedVideoEmotion?: any) {
    console.log('📤 Sending audio with video emotion:', capturedVideoEmotion);
    
    this.addMessage('🎤 Voice message', true);
    
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    
    if (capturedVideoEmotion && capturedVideoEmotion.emotion) {
      formData.append('video_emotion', JSON.stringify(capturedVideoEmotion));
    }
    
    this.isTyping = true;

    this.http.post<any>('http://127.0.0.1:8000/predict-emotion-with-video', formData)
      .subscribe({
        next: (res) => {
          console.log('✅ Audio response received');
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
            // ⭐ Save with session_id
            this.saveChatToHistory(displayText, botResponse, emotion, res.emotion_confidence || 0);
            
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
    return `emotion-${emotion?.toLowerCase() || 'neutral'}`;
  }

  cancelRecording() {
    this.clearPreviousRecording();
  }

  // ==================== SAVE CHAT - WITH SESSION ====================
  private saveChatToHistory(userMessage: string, botResponse: string, emotion: string, confidence: number) {
    if (!this.currentUserId || !this.currentSessionId) {
      console.error('❌ Missing user_id or session_id');
      return;
    }

    const payload = {
      user_id: this.currentUserId,
      session_id: this.currentSessionId,  // ⭐ Include session_id
      user_message: userMessage,
      bot_response: botResponse,
      emotion: emotion,
      emotion_confidence: confidence
    };
   
    this.http.post<any>('http://127.0.0.1:8000/save-chat', payload)
      .subscribe({
        next: (res) => {
          console.log('✓ Chat saved to session:', res.session_id);
        },
        error: (err) => {
          console.error('Error saving chat:', err);
        }
      });
  }

  // ==================== NEW CHAT ====================
  newChat() {
    // ⭐ Reset everything for new session
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

    // ⭐ Generate new session ID
    this.currentSessionId = this.generateSessionId();
    localStorage.setItem('currentSessionId', this.currentSessionId);

    console.log('✓ New chat started with session:', this.currentSessionId);
    this.cdr.detectChanges();
  }
}
