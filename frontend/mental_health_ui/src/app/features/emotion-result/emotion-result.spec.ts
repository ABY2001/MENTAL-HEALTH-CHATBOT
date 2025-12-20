import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EmotionResult } from './emotion-result';

describe('EmotionResult', () => {
  let component: EmotionResult;
  let fixture: ComponentFixture<EmotionResult>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EmotionResult]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EmotionResult);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
