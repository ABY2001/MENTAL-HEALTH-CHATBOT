import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SafetyMessage } from './safety-message';

describe('SafetyMessage', () => {
  let component: SafetyMessage;
  let fixture: ComponentFixture<SafetyMessage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SafetyMessage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SafetyMessage);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
