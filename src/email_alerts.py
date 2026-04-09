"""
HTML email alert template generator.

Produces a clean HTML email body from a list of Alert objects.
Actual sending is not implemented here — this module only generates the template.
"""
from src.alerts import Alert

_SEVERITY_COLORS = {
    "critical": "#dc3545",
    "warning":  "#ffc107",
    "info":     "#0dcaf0",
}

_SEVERITY_TEXT_COLORS = {
    "critical": "#ffffff",
    "warning":  "#212529",
    "info":     "#212529",
}


def format_alert_email(alerts: list[Alert]) -> str:
    """
    Generate an HTML email body summarizing today's alerts.

    Parameters
    ----------
    alerts : list of Alert objects (typically from generate_daily_alerts())

    Returns
    -------
    str — complete HTML document ready to send as email body
    """
    count_critical = sum(1 for a in alerts if a.severity == "critical")
    count_warning  = sum(1 for a in alerts if a.severity == "warning")
    count_info     = sum(1 for a in alerts if a.severity == "info")

    rows = ""
    for a in alerts:
        bg    = _SEVERITY_COLORS.get(a.severity, "#6c757d")
        fg    = _SEVERITY_TEXT_COLORS.get(a.severity, "#ffffff")
        rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #e9ecef;width:90px;vertical-align:top;">
            <span style="display:inline-block;padding:3px 10px;border-radius:4px;
                         background:{bg};color:{fg};font-size:11px;font-weight:bold;
                         text-transform:uppercase;letter-spacing:0.5px;">
              {a.severity}
            </span>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #e9ecef;vertical-align:top;">
            <strong style="color:#212529;">{a.title}</strong><br>
            <span style="color:#6c757d;font-size:13px;line-height:1.5;">
              {a.description}
            </span>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #e9ecef;
                     color:#198754;font-size:13px;vertical-align:top;width:220px;">
            {a.recommended_action}
          </td>
        </tr>"""

    if not alerts:
        body_content = """
        <p style="color:#6c757d;text-align:center;padding:32px;">
          No alerts today — everything looks good!
        </p>"""
    else:
        body_content = f"""
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;border:1px solid #dee2e6;border-radius:6px;
                      overflow:hidden;font-family:Arial,sans-serif;font-size:14px;">
          <thead>
            <tr style="background:#f8f9fa;">
              <th style="padding:10px 12px;text-align:left;color:#495057;font-size:12px;
                         text-transform:uppercase;letter-spacing:0.5px;">Severity</th>
              <th style="padding:10px 12px;text-align:left;color:#495057;font-size:12px;
                         text-transform:uppercase;letter-spacing:0.5px;">Alert</th>
              <th style="padding:10px 12px;text-align:left;color:#495057;font-size:12px;
                         text-transform:uppercase;letter-spacing:0.5px;">Recommended Action</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""

    from datetime import date
    today_str = date.today().strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f8f9fa;font-family:Arial,sans-serif;">
  <div style="max-width:700px;margin:24px auto;background:white;border-radius:8px;
              box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">

    <div style="background:#1a1a2e;padding:24px 32px;">
      <h1 style="color:white;margin:0;font-size:20px;font-weight:600;">
        Hotel Analytics — Daily Alert Summary
      </h1>
      <p style="color:#adb5bd;margin:6px 0 0;font-size:14px;">{today_str}</p>
    </div>

    <div style="padding:24px 32px;">
      <p style="color:#495057;font-size:15px;margin:0 0 20px;">
        You have <strong>{len(alerts)} alert(s)</strong> today:
        <span style="color:#dc3545;">{count_critical} critical</span>,
        <span style="color:#856404;">{count_warning} warning</span>,
        <span style="color:#0a58ca;">{count_info} info</span>.
      </p>
      {body_content}
    </div>

    <div style="padding:16px 32px;background:#f8f9fa;border-top:1px solid #e9ecef;">
      <p style="color:#adb5bd;font-size:12px;margin:0;text-align:center;">
        Powered by Hotel Analytics Platform &nbsp;·&nbsp;
        <a href="#" style="color:#adb5bd;">Unsubscribe</a>
      </p>
    </div>

  </div>
</body>
</html>"""
