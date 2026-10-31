import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kathak AI Guru • कथक नृत्य मुद्रा परिशोधन",
  description: "AI-Powered Indian Classical Kathak Dance Pose Alignment, Kinematic Analysis & Visual Correction",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col font-sans bg-[#0c0205] text-[#fdf8eb] selection:bg-amber-500/30 selection:text-amber-200">
        {children}
      </body>
    </html>
  );
}
