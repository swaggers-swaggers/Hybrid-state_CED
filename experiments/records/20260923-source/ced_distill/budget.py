"""Predeclared throughput sizing and pair-safe wall-clock limits; no metric tuning."""
import time

class TokenBudget:
    def __init__(self,maximum,seconds,clock=time.monotonic,sample=1024,safety=.65):
        if maximum<1 or seconds<=0:raise ValueError('Invalid time budget')
        self.maximum=maximum;self.target=maximum;self.seconds=seconds;self.clock=clock
        self.started=clock();self.sample=sample;self.safety=safety
        self.measured_rate=None;self.reason=None
    def boundary(self,positions):
        elapsed=self.clock()-self.started
        if self.measured_rate is None and positions>=min(self.sample,self.maximum):
            self.measured_rate=positions/max(elapsed,1e-9)
            additional=int(self.measured_rate*max(0,self.seconds-elapsed)*self.safety)
            self.target=min(self.maximum,positions+additional)
        if elapsed>=self.seconds:self.reason='WALL_TIME_LIMIT'
        elif positions>=self.target:self.reason='TOKEN_TARGET'
        return self.reason is not None
    def report(self):
        return dict(maximum_tokens=self.maximum,chosen_tokens=self.target,actual_seconds=self.clock()-self.started,
                    training_seconds_limit=self.seconds,initial_tokens_per_second=self.measured_rate,
                    sizing_safety_factor=self.safety,stop_reason=self.reason)
