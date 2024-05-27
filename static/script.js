document.addEventListener('DOMContentLoaded', function() {
    if (typeof latestDates !== 'undefined' && typeof latestPrices !== 'undefined') {
        const ctx = document.getElementById('priceChart').getContext('2d');
        const priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: latestDates,
                datasets: [{
                    label: 'Stock Price',
                    data: latestPrices,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (USD)'
                        },
                        beginAtZero: false
                    }
                }
            }
        });
    }
});
