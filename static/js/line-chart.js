// let chartData = {
//   labels: [
//     "01/05/2023",
//     "02/05/2023",
//     "03/05/2023",
//     "04/05/2023",
//     "05/05/2023",
//     "06/05/2023",
//     "07/05/2023",
//   ],
//   description: "Số lượng quan điểm",
//   dataset: [
//     {
//       label: "Positive",
//       data: [10, 20, 30, 30, 50, 60, 100],
//     },
//     {
//       label: "Negative",
//       data: [50, 20, 20, 10, 35, 5, 5],
//     },
//     {
//       label: "Neutral",
//       data: [100, 58, 50, 30, 60, 60, 97],
//     },
//   ],
// };

function displayLineChart(dataChart) {
  chartContent = document.getElementById("line-chart-content");

  const data = {
    labels: dataChart.labels,
    datasets: [
      {
        label: dataChart.dataset[0].label,
        data: dataChart.dataset[0].data,
        fill: false,
        borderColor: "rgb(152, 216, 170)",
        tension: 0,
        line: {
          backgroundColor: "rgb(152, 216, 170)",
        },
      },
      {
        label: dataChart.dataset[1].label,
        data: dataChart.dataset[1].data,
        fill: false,
        borderColor: "rgb(255, 109, 96)",
        tension: 0,
      },
      {
        label: dataChart.dataset[2].label,
        data: dataChart.dataset[2].data,
        fill: false,
        borderColor: "rgb(247, 208, 96)",
        tension: 0,
      },
    ],
  };
  const config = {
    type: "line",
    data: data,
    option: {
      maintainAspectRatio: true,
      responsive: true,
      plugins: {
        tooltip: {
          mode: "index",
          intersect: false,
        },
        title: {
          display: true,
          text: "Chart.js Line Chart",
        },
      },
      hover: {
        mode: "index",
        intersec: false,
      },
      scales: {
        y: {
          title: {
            display: true,
            text: "Value",
          },
          min: 0,
          max: 500,
          ticks: {
            // forces step size to be 50 units
            stepSize: 10,
          },
        },
      },
    },
  };
  new Chart(chartContent, config);
}
