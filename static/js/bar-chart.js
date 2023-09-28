function displayBarChart(chartData) {
  console.log("Something");
  const chartContent = document.getElementById("chart-content");

  new Chart(chartContent, {
    type: "bar",
    data: {
      labels: chartData.labels,
      datasets: [
        {
          label: "Item", // Mô tả thuộc tính được biểu diễn bởi biểu đồ (VD: Độ chính xác, xác suất, dân số,...)
          data: chartData.data, // Dữ liệu biểu diễn
          backgroundColor: [
            "rgb(152, 216, 170)",
            "rgb(255, 109, 96)",
            "rgb(247, 208, 96)",
          ],
          barPercentage: 0.5,
          barThickness: 50,
          maxBarThickness: 50,
          minBarLength: 2,
        },
      ],
    },
    options: {
      borderWidth: 0,
      borderRadius: 2,
      hoverBorderWidth: 0,
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}
