// let chartData = {
//   labels: ["Positive", "Negative", "Neutral"],
//   data: [14.55, 20.63, 64.82],
// };

function displayDoughnutChart(chartData) {
  const chartContent = document.getElementById("chart-content");

  new Chart(chartContent, {
    type: "doughnut",
    data: {
      labels: chartData.labels,
      datasets: [
        {
          label: "Xác suất ", // Mô tả thuộc tính được biểu diễn bởi biểu đồ (VD: Độ chính xác, xác suất, dân số,...)
          data: chartData.data, // Dữ liệu biểu diễn
          backgroundColor: [
            "rgb(152, 216, 170)",
            "rgb(255, 109, 96)",
            "rgb(247, 208, 96)",
          ],
        },
      ],
    },
    options: {
      borderWidth: 5,
      borderRadius: 2,
      hoverBorderWidth: 0,
      responsive: true,
      elements: {
        arc: {},
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: {
            boxWidth: 20,
            boxHeight: 20,
            font: {
              size: 16,
            },
          },
        },
      },
    },
  });
}
