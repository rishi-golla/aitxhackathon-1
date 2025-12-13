interface AnalyticsRowProps {
  children: React.ReactNode
}

export default function AnalyticsRow({ children }: AnalyticsRowProps) {
  return (
    <section className="grid grid-cols-3 gap-6">
      {children}
    </section>
  )
}
