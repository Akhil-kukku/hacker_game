# Submit Labeled Feedback Samples for Model Adaptation
# This script sends 5 pre-configured labeled samples (3 attacks, 2 benign)

$baseUrl = "http://localhost:8000"
$endpoint = "$baseUrl/order/feedback"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  SUBMITTING LABELED FEEDBACK FOR MODEL ADAPTATION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Define 5 labeled samples (3 attacks, 2 benign)
$feedbackSamples = @(
    @{
        label = "attack"
        flow = @{
            src_ip = "192.168.1.105"
            dst_ip = "10.0.0.50"
            src_port = 54321
            dst_port = 22
            protocol = "TCP"
            packet_count = 15000
            byte_count = 2500000
            duration = 300.5
        }
    },
    @{
        label = "attack"
        flow = @{
            src_ip = "192.168.1.200"
            dst_ip = "10.0.0.100"
            src_port = 49876
            dst_port = 3389
            protocol = "TCP"
            packet_count = 25000
            byte_count = 5000000
            duration = 600.2
        }
    },
    @{
        label = "attack"
        flow = @{
            src_ip = "172.16.0.50"
            dst_ip = "10.0.0.75"
            src_port = 60000
            dst_port = 1433
            protocol = "TCP"
            packet_count = 30000
            byte_count = 7500000
            duration = 450.8
        }
    },
    @{
        label = "benign"
        flow = @{
            src_ip = "192.168.1.50"
            dst_ip = "93.184.216.34"
            src_port = 54320
            dst_port = 443
            protocol = "HTTPS"
            packet_count = 150
            byte_count = 75000
            duration = 5.2
        }
    },
    @{
        label = "benign"
        flow = @{
            src_ip = "192.168.1.75"
            dst_ip = "151.101.1.140"
            src_port = 54321
            dst_port = 80
            protocol = "HTTP"
            packet_count = 80
            byte_count = 40000
            duration = 2.5
        }
    }
)

$successCount = 0

foreach ($sample in $feedbackSamples) {
    $attackLabel = if ($sample.label -eq "attack") { "ATTACK" } else { "BENIGN" }
    $color = if ($sample.label -eq "attack") { "Red" } else { "Green" }
    $isAttack = if ($sample.label -eq "attack") { $true } else { $false }
    
    Write-Host "Submitting $attackLabel sample: $($sample.flow.src_ip) -> $($sample.flow.dst_ip):$($sample.flow.dst_port)" -ForegroundColor $color
    
    try {
        # API expects flat fields, not nested flow object
        $body = @{
            src_ip = $sample.flow.src_ip
            dst_ip = $sample.flow.dst_ip
            src_port = $sample.flow.src_port
            dst_port = $sample.flow.dst_port
            protocol = $sample.flow.protocol
            packet_count = $sample.flow.packet_count
            byte_count = $sample.flow.byte_count
            duration = $sample.flow.duration
            flags = ""
            is_attack = $isAttack
        } | ConvertTo-Json -Depth 2
        
        $response = Invoke-RestMethod -Uri $endpoint -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop
        Write-Host "  Success - Buffer size: $($response.buffer_size)" -ForegroundColor Gray
        $successCount++
        Start-Sleep -Milliseconds 500
    }
    catch {
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Feedback submission complete: $successCount/$($feedbackSamples.Count) successful" -ForegroundColor Cyan
Write-Host "Model will adapt when buffer reaches threshold (1000 samples)" -ForegroundColor Gray
Write-Host ""
