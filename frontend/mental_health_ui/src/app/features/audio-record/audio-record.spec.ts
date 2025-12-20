import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AudioRecord } from './audio-record';

describe('AudioRecord', () => {
  let component: AudioRecord;
  let fixture: ComponentFixture<AudioRecord>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AudioRecord]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AudioRecord);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
