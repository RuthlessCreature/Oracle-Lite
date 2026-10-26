"""Numerical hardware compatibility; unknown electrical/mechanical data stay unknown."""
import math


def panel_candidates(cat,required):
    rows=[]
    for item in cat['items']:
        s=item.get('spec') or {}
        if item['kind']!='light' or s.get('type')!='backlight':continue
        if not s.get('active_mm') or not s.get('body_mm'):continue
        active=s['active_mm'];rotations=[angle for angle,size in [(0,active),(90,active[::-1])] if all(a>=b for a,b in zip(size,required))]
        if rotations:rows.append(dict(item_id=item['id'],rotations_deg=rotations,active_mm=active,body_mm=s['body_mm'],source_url=item['source_url'],effect_verified=False))
    return rows


def controller_check(light,controller,*,pulse_ms=None,duty_pct=None):
    l=light.get('spec') or {};c=controller.get('spec') or {};missing=[];violations=[];evidence=[]
    voltage=l.get('voltage_v');current=l.get('current_a')
    if current is None and voltage and l.get('power_w'):
        current=l['power_w']/voltage;evidence.append('current estimated as rated power / rated voltage; driver compatibility remains unverified')
    allowed=c.get('output_voltage_v')
    if voltage is None or allowed is None:missing.append('light voltage or controller output voltage')
    elif isinstance(allowed,list) and not (len(allowed)==2 and allowed[0]<=voltage<=allowed[1]):violations.append('output_voltage_mismatch')
    elif not isinstance(allowed,list) and voltage!=allowed:violations.append('output_voltage_mismatch')
    max_current=c.get('output_current_a')
    if current is None or max_current is None:missing.append('rated light current or controller channel current')
    elif current>max_current:violations.append('channel_current_exceeded')
    power=l.get('power_w');limit=c.get('output_power_w')
    if power is not None and limit is not None and power>limit:violations.append('output_power_exceeded')
    for actual,key in [(pulse_ms,'max_pulse_ms'),(duty_pct,'max_duty_pct')]:
        if actual is not None:
            if not math.isfinite(actual) or actual<0:raise ValueError('Invalid illumination timing')
            if l.get(key) is None:missing.append(key)
            elif actual>l[key]:violations.append(key+'_exceeded')
    return dict(light_id=light['id'],controller_id=controller['id'],violations=violations,missing=missing,
                status='rejected' if violations else ('incomplete_specs' if missing else 'numeric_ranges_supported'),
                evidence=evidence,wiring_verified=False,connector_verified=False,ready_for_installation=False)
