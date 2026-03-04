import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EmotionWidget } from './emotion-widget';

describe('EmotionWidget', () => {
  let component: EmotionWidget;
  let fixture: ComponentFixture<EmotionWidget>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EmotionWidget]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EmotionWidget);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
