# Generate Load for Latency Testing
# This script submits 200 random network flows to measure processing latency

$baseUrl = "http://localhost:8000"
$endpoint = "$baseUrl/flows"
$flowCount = 200
$batchSize = 10

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  GENERATING LOAD: Submitting $flowCount flows for latency testing" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$ports = @(80, 443, 22, 53, 25, 3306, 3389, 1433)
$protocols = @("TCP", "UDP", "HTTP", "HTTPS")

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$successCount = 0
$batch = @()

try {
    for ($i = 1; $i -le $flowCount; $i++) {
        $flow = @{
            src_ip = "192.168.1.$((Get-Random -Minimum 1 -Maximum 254))"
            dst_ip = "10.0.0.$((Get-Random -Minimum 1 -Maximum 254))"
            src_port = Get-Random -Minimum 10000 -Maximum 65000
            dst_port = $ports[(Get-Random -Minimum 0 -Maximum $ports.Count)]
            protocol = $protocols[(Get-Random -Minimum 0 -Maximum $protocols.Count)]
            packet_count = Get-Random -Minimum 10 -Maximum 5000
            byte_count = Get-Random -Minimum 1000 -Maximum 500000
            duration = [math]::Round((Get-Random -Minimum 1 -Maximum 300) + (Get-Random) / 10, 2)
            flags = ""
        }
        
        $batch += $flow
        
        # Send batch when full or at end
        if ($batch.Count -eq $batchSize -or $i -eq $flowCount) {
            try {
                $body = $batch | ConvertTo-Json -Depth 2
                $response = Invoke-RestMethod -Uri $endpoint -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop
                $successCount += $batch.Count
                $batch = @()
                
                if ($i % 50 -eq 0) {
                    Write-Host "  Progress: $i/$flowCount flows sent" -ForegroundColor Gray
                }
            }
            catch {
                # Continue on batch failures
                $batch = @()
            }
        }
    }
}
catch {
    Write-Host "Failed to send flows" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

$stopwatch.Stop()
$elapsed = $stopwatch.ElapsedMilliseconds
$avgPerFlow = [math]::Round($elapsed / $flowCount, 2)

Write-Host ""
Write-Host "Load generation complete: $successCount/$flowCount flows processed" -ForegroundColor Green
Write-Host "Total time: $elapsed ms" -ForegroundColor Gray
Write-Host "Average per flow: $avgPerFlow ms" -ForegroundColor Gray
Write-Host ""
Write-Host "Tip: Run this script 3-5 times to generate sufficient load for latency testing" -ForegroundColor Cyan
Write-Host ""
