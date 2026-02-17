// audio-record.component.ts - WITH CHAT HISTORY INTEGRATED

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
  imports: [CommonModule, FormsModule],
  templateUrl: './audio-record.html',
  styleUrl: './audio-record.css'
})
export class AudioRecord implements OnDestroy, AfterViewChecked {
  @ViewChild('messagesContainer') messagesContainer!: ElementRef;
  
  // Current user
  currentUserId: number | null = null;
  
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
  
  // Chat history state
  showChatHistory = false;
  chatHistory: ChatSession[] = [];
  chatHistoryLoading = false;
  
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
  audioPreviewDuration = '0:00'; 
  
  // Emotion state
  currentEmotion: string | null = null;
  emotionStatus = '😊 Ready to chat';
  
  private shouldScrollToBottom = false;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef 
  ) {
    // Get user_id from localStorage (set during login)
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

  onKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
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
          console.log(`Loaded ${this.chatHistory.length} chat messages`);
          this.cdr.detectChanges();
        },
        error: (err) => {
          console.error('Error loading chat history:', err);
          this.chatHistoryLoading = false;
          alert('Failed to load chat history');
        }
      });
  }

  loadPreviousChat(chat: ChatSession) {
    // Close history panel
    this.showChatHistory = false;

    // Clear current messages and load the selected chat
    this.messages = [
      {
        text: "Loading previous conversation...",
        isUser: false,
        time: this.getCurrentTime()
      }
    ];

    // Add the selected chat to messages
    const timestamp = new Date(chat.created_at).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });

    this.messages = [];
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

    const confirm = window.confirm('Are you sure you want to delete all chat history? This cannot be undone.');
    if (!confirm) return;

    this.http.delete<any>(`http://127.0.0.1:8000/delete-chat-session/${this.currentUserId}`)
      .subscribe({
        next: (res) => {
          console.log(`Deleted ${res.messages_deleted} messages`);
          this.chatHistory = [];
          this.messages = [
            {
              text: "Hello! I'm here to support you. How are you feeling today?",
              isUser: false,
              time: this.getCurrentTime()
            }
          ];
          alert('Chat history deleted');
          this.cdr.detectChanges();
        },
        error: (err) => {
          console.error('Error deleting chat history:', err);
          alert('Failed to delete chat history');
        }
      });
  }

  closeChatHistory() {
    this.showChatHistory = false;
  }

  // ==================== TEXT MESSAGE ====================
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
            
            // Save chat to history
            this.saveChatToHistory(text, botResponse, emotion, res.confidence);
            
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

  // ==================== AUDIO ====================
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
          console.log('Bot response:', botResponse);
          
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
            
            // Save chat to history
            const displayText = transcription || '🎤 Voice message';
            this.saveChatToHistory(displayText, botResponse, emotion, res.confidence);
            
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

  // ==================== SAVE CHAT TO HISTORY ====================
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
          console.log('✓ Chat saved to history');
        },
        error: (err) => {
          console.error('Error saving chat to history:', err);
        }
      });
  }
  newChat() {
  const confirm = window.confirm('Start a new conversation? Current messages will be cleared.');
  if (!confirm) return;

  // Reset messages to initial state
  this.messages = [
    {
      text: "Hello! I'm here to support you. How are you feeling today? You can type or speak to me.",
      isUser: false,
      time: this.getCurrentTime()
    }
  ];

  // Reset input
  this.messageInput = '';
  this.currentEmotion = null;
  this.emotionStatus = '😊 Ready to chat';

  // Clear any open modals
  this.showAudioPreview = false;
  this.showChatHistory = false;
  this.isRecording = false;

  // Stop any active recording
  if (this.recordingInterval) {
    clearInterval(this.recordingInterval);
    this.recordingInterval = null;
  }

  console.log('✓ New chat started');
  this.cdr.detectChanges();
}
deleteIndividualChat(chatId: number | undefined, chatPreview: string) {
  // ✅ Check if chatId exists
  if (!chatId) {
    alert('Error: Chat ID not found');
    return;
  }

  const confirm = window.confirm(`Delete this chat?\n"${chatPreview}..."`);
  if (!confirm) return;

  this.http.delete<any>(`http://127.0.0.1:8000/delete-chat-message/${chatId}`)
    .subscribe({
      next: (res) => {
        this.chatHistory = this.chatHistory.filter(chat => chat.id !== chatId);
        alert('Chat deleted successfully');
        this.cdr.detectChanges();
      },
      error: (err) => {
        alert('Failed to delete chat');
      }
    });
}
}