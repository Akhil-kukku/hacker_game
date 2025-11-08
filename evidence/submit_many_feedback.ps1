# Submit Multiple Feedback Batches
# This script submits enough labeled samples to trigger model retraining (50+ samples)

$baseUrl = "http://localhost:8000"
$endpoint = "$baseUrl/order/feedback"
$targetSamples = 60  # Exceed threshold of 50

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  SUBMITTING $targetSamples LABELED SAMPLES FOR MODEL RETRAINING" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$successCount = 0
$attackCount = 0
$benignCount = 0

# Define patterns for attacks and benign traffic
$attackPorts = @(22, 23, 3389, 1433, 3306, 5432)  # SSH, Telnet, RDP, SQL, MySQL, PostgreSQL
$benignPorts = @(80, 443, 53, 25, 143, 993)      # HTTP, HTTPS, DNS, SMTP, IMAP, IMAPS

for ($i = 1; $i -le $targetSamples; $i++) {
    # Alternate between attack and benign (roughly 50/50 split)
    $isAttack = ($i % 2 -eq 1)
    
    if ($isAttack) {
        # Attack pattern: High packet/byte counts, longer duration, suspicious ports
        $flow = @{
            src_ip = "192.168.1.$((Get-Random -Minimum 100 -Maximum 200))"
            dst_ip = "10.0.0.$((Get-Random -Minimum 1 -Maximum 100))"
            src_port = Get-Random -Minimum 50000 -Maximum 65000
            dst_port = $attackPorts[(Get-Random -Minimum 0 -Maximum $attackPorts.Count)]
            protocol = "TCP"
            packet_count = Get-Random -Minimum 10000 -Maximum 50000
            byte_count = Get-Random -Minimum 1000000 -Maximum 10000000
            duration = [math]::Round((Get-Random -Minimum 100 -Maximum 1000) + (Get-Random), 2)
            flags = "S"
        }
        $label = "ATTACK"
        $color = "Red"
        $attackCount++
    } else {
        # Benign pattern: Normal packet/byte counts, short duration, common ports
        $flow = @{
            src_ip = "192.168.1.$((Get-Random -Minimum 1 -Maximum 100))"
            dst_ip = "8.8.$((Get-Random -Minimum 4 -Maximum 8)).8"
            src_port = Get-Random -Minimum 50000 -Maximum 65000
            dst_port = $benignPorts[(Get-Random -Minimum 0 -Maximum $benignPorts.Count)]
            protocol = if ($flow.dst_port -eq 53) { "UDP" } else { "TCP" }
            packet_count = Get-Random -Minimum 50 -Maximum 500
            byte_count = Get-Random -Minimum 5000 -Maximum 100000
            duration = [math]::Round((Get-Random -Minimum 1 -Maximum 30) + (Get-Random), 2)
            flags = ""
        }
        $label = "BENIGN"
        $color = "Green"
        $benignCount++
    }
    
    try {
        $body = @{
            src_ip = $flow.src_ip
            dst_ip = $flow.dst_ip
            src_port = $flow.src_port
            dst_port = $flow.dst_port
            protocol = $flow.protocol
            packet_count = $flow.packet_count
            byte_count = $flow.byte_count
            duration = $flow.duration
            flags = $flow.flags
            is_attack = $isAttack
        } | ConvertTo-Json -Depth 2
        
        $response = Invoke-RestMethod -Uri $endpoint -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop
        $successCount++
        
        if ($i % 10 -eq 0) {
            Write-Host "  Progress: $i/$targetSamples samples | Buffer: $($response.buffer_size)" -ForegroundColor Gray
        }
        
        # Brief delay to avoid overwhelming server
        Start-Sleep -Milliseconds 50
    }
    catch {
        Write-Host "  Error on sample $i : $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  FEEDBACK SUBMISSION COMPLETE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Submitted:  $successCount/$targetSamples samples" -ForegroundColor Green
Write-Host "  Attacks:  $attackCount" -ForegroundColor Red
Write-Host "  Benign:   $benignCount" -ForegroundColor Green
Write-Host ""
Write-Host "Model should have retrained when buffer reached 50 samples" -ForegroundColor Yellow
Write-Host "Run capture_evidence.ps1 to capture new metrics with improvements" -ForegroundColor Cyan
Write-Host ""
