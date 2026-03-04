import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-emotion-widget',
  imports: [CommonModule],
  templateUrl: './emotion-widget.html',
  styleUrl: './emotion-widget.css',
})
export class EmotionWidget {
 @Input() currentEmotion: string | null = null;
 @Input() isVideoActive = false;
 @Input() isRecording = false;

  getEmoji(emotion: string | null): string {
    const emotionEmojis: { [key: string]: string } = {
      happy: '😊',
      sad: '😔',
      angry: '😠',
      fearful: '😰',
      neutral: '😐',
      calm: '😌',
      disgust: '😖',
      surprised: '😲',
      fear: '😨',
      surprise: '😲'
    };
    return emotionEmojis[emotion?.toLowerCase() || 'neutral'] || '💙';
  }
  
  ngOnInit() {

  }
}
