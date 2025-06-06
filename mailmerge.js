/**
 * Sends personalised emails based on Data and Bodies sheets.
 * Quota: Gmail free = 500/day | Workspace = 2 000/day.
 */
function sendAllEmails() {
  const ss      = SpreadsheetApp.getActive();
  const dataS   = ss.getSheetByName('Data');
  const bodyS   = ss.getSheetByName('Bodies');
  
  // --- Build a {BodyID: BodyText} map ---
  const bodies = bodyS.getRange(1, 1, 8, 2).getValues()
    .reduce((m, [id, text]) => (m[id] = text, m), {});
  
  // --- Grab every data row ---
  const rows = dataS.getRange(2, 1, dataS.getLastRow() - 1, 3).getValues();
  
  const subject = 'Quick update';                         // <-- customise once
  const signature = '\n\nBest,\nHer Name';                // <-- customise once
  
  rows.forEach(([email, name, bodyID], i) => {
    const body = `Hi ${name},\n\n${bodies[bodyID]}${signature}`;
    GmailApp.sendEmail(email, subject, body);
    
    // Optional: throttle to 500/day on free Gmail
    if (Session.getActiveUser().getEmail().endsWith('@gmail.com') &&
        (i + 1) % 500 === 0) {
      Logger.log('500 sent… waiting 24 h');
      Utilities.sleep(24 * 60 * 60 * 1000);
    }
  });
  
  SpreadsheetApp.getUi().alert(`Done! Sent ${rows.length} emails.`);
}