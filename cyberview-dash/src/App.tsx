import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import LiveDetection from "./pages/LiveDetection";
import LiveAnalysis from "./pages/LiveAnalysis";
import TrafficAnalytics from "./pages/TrafficAnalytics";
import Explainability from "./pages/Explainability";
import Prevention from "./pages/Prevention";
import Logs from "./pages/Logs";
import About from "./pages/About";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout><Dashboard /></Layout>} />
          <Route path="/live-detection" element={<Layout><LiveDetection /></Layout>} />
          <Route path="/live-analysis" element={<Layout><LiveAnalysis /></Layout>} />
          <Route path="/traffic-analytics" element={<Layout><TrafficAnalytics /></Layout>} />
          <Route path="/explainability" element={<Layout><Explainability /></Layout>} />
          <Route path="/prevention" element={<Layout><Prevention /></Layout>} />
          <Route path="/logs" element={<Layout><Logs /></Layout>} />
          <Route path="/about" element={<Layout><About /></Layout>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
